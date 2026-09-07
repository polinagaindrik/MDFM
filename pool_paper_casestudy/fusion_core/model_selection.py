"""
Generic AIC/BIC model-comparison engine.

Moved from ``pool_paper_casestudy/model_selection_hill_vs_mm.py``. Compares
any set of already-fitted models under a shared i.i.d.-Gaussian-residual
assumption, using AIC/AICc/BIC (Burnham & Anderson) -- none of that math is
specific to the Hill-vs-Michaelis-Menten comparison it was originally
written for, or to this case study at all.

Only ``evaluate_model`` needed generalizing: it used to call the
`cost(param, calibr_setup, jac_spasity)` function imported from
`pool_paper_casestudy/do_local_optim.py` directly. It now takes `cost_func`
as an explicit argument, the same convention used throughout
`fusion_core` (see `fusion_core.pest.calculate_model_params` and
`fusion_core.likelihood`).

    sigma_hat^2 = RSS/n            (plain MLE, no degrees-of-freedom correction)
    AIC = -2logL + 2*(n_params+1)
    BIC = -2logL + (n_params+1)*ln(n)

RSS/n = cost_opt (`cost_arithmetic_mean` already returns the mean, so
RSS = cost_opt * n_data). The "+1" in (n_params+1) accounts for sigma^2
itself being an estimated parameter, standard convention for Gaussian
least-squares AIC/BIC.

Lower AIC/BIC = preferred model. Delta-AIC/BIC and (for AIC) Akaike weights
are reported for interpretability:
    - Delta < 2   : models are essentially indistinguishable
    - Delta 2-10  : some to strong support for the lower-scoring model
    - Delta > 10  : the higher-scoring model is effectively ruled out
"""

import numpy as np
import pandas as pd

from . import likelihood as _likelihood


# ----------------------------------------------------------------------
# AIC / BIC computation
# ----------------------------------------------------------------------

def compute_aic_bic(cost_opt, n_data, n_params):
    """
    cost_opt  : cost_arithmetic_mean at this model's own optimum (RSS/n_data)
    n_data    : number of individual residual terms (from count_data_points)
    n_params  : number of free model parameters (NOT including sigma^2)

    sigma_hat2 : plain MLE estimate of the residual variance, RSS/n_data
                 (no n_params correction -- see
                 `fusion_core.likelihood.estimate_profile_scale()` for the
                 bias-corrected variant used elsewhere in this project).
    k          : n_params + 1, since sigma^2 is estimated via MLE
                 alongside the model parameters (standard AIC/BIC
                 convention for Gaussian least-squares).
    AICc       : small-sample-corrected AIC (Burnham & Anderson),
                 AICc = AIC + 2k(k+1)/(n_data-k-1). Recommended over
                 plain AIC whenever n_data/k < 40 (see the n/k rule in
                 `compare_models` below); undefined/unstable if
                 n_data <= k+1.

    Returns dict with RSS, sigma_hat2, neg2logL, k, AIC, AICc, BIC.
    """
    RSS = cost_opt * n_data
    if RSS <= 0:
        raise ValueError(f"RSS={RSS} <= 0; can't take log. Check cost_opt/n_data.")

    sigma_hat2 = RSS / n_data  # == cost_opt, written explicitly for clarity
    neg2logL = n_data * np.log(sigma_hat2) + n_data * (np.log(2 * np.pi) + 1)
    k = n_params + 1  # +1 for sigma^2, estimated via MLE alongside the model params

    AIC = neg2logL + 2 * k
    BIC = neg2logL + k * np.log(n_data)

    if n_data - k - 1 <= 0:
        raise ValueError(
            f"n_data-k-1={n_data-k-1} <= 0; AICc is undefined here (too many "
            "parameters relative to the data -- the model can't be reliably "
            "compared at all, let alone with the small-sample correction)."
        )
    AICc = AIC + (2 * k * (k + 1)) / (n_data - k - 1)

    return {
        "cost_opt": cost_opt,
        "RSS": RSS,
        "n_data": n_data,
        "n_params": n_params,
        "sigma_hat2": sigma_hat2,
        "k": k,
        "neg2logL": neg2logL,
        "AIC": AIC,
        "AICc": AICc,
        "BIC": BIC,
    }


def compare_models(model_results, verbose=True):
    """
    model_results : dict {model_label: {"cost_opt":..., "n_data":..., "n_params":...}}
                     (n_data should be the same across models if they were fit
                     to the same dataset -- a mismatch is flagged below)

    Returns a DataFrame with sigma_hat2, -2logL, AIC, BIC, delta-AIC,
    delta-BIC, and Akaike weights for each model, sorted by AIC (best first).
    """
    rows = []
    for label, r in model_results.items():
        stats = compute_aic_bic(r["cost_opt"], r["n_data"], r["n_params"])
        stats["model"] = label
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("model")

    n_data_vals = df["n_data"].unique()
    if len(n_data_vals) > 1 and verbose:
        print(
            f"WARNING: n_data differs across models ({dict(df['n_data'])}). "
            "AIC/BIC are only directly comparable if all models were fit to "
            "the exact same data points -- check your calibr_setup/data_array "
            "are consistent across models before trusting this comparison."
        )

    df = df.sort_values("AIC")
    df["delta_AIC"] = df["AIC"] - df["AIC"].min()
    df["delta_AICc"] = df["AICc"] - df["AICc"].min()
    df["delta_BIC"] = df["BIC"] - df["BIC"].min()

    # Akaike weights: relative likelihood of each model given the set
    rel_likelihood = np.exp(-0.5 * df["delta_AIC"])
    df["akaike_weight"] = rel_likelihood / rel_likelihood.sum()

    # n/k rule of thumb (Burnham & Anderson): AICc recommended when n/k < 40
    df["n_over_k"] = df["n_data"] / df["k"]
    low_ratio = df[df["n_over_k"] < 40]
    if len(low_ratio) and verbose:
        ratios = ", ".join(f"{m}: n/k={v:.3g}" for m, v in low_ratio["n_over_k"].items())
        print(
            f"NOTE: n_data/k < 40 for {ratios} -- plain AIC is unreliable at this "
            "sample-size-to-parameter ratio; prefer AICc for these models "
            "(Burnham & Anderson)."
        )

    if verbose:
        print("\nModel comparison (sorted by AIC, best first):")
        print(df[["sigma_hat2", "neg2logL", "n_data", "k", "n_over_k", "AIC", "delta_AIC",
                   "AICc", "delta_AICc", "akaike_weight", "BIC", "delta_BIC"]]
              .rename(columns={"neg2logL": "-2logL"})
              .to_string(float_format=lambda x: f"{x:.6g}"))

        best_aic = df.index[0]
        best_aicc = df.sort_values("AICc").index[0]
        best_bic = df.sort_values("BIC").index[0]
        print(f"\nAIC prefers: {best_aic}")
        print(f"AICc prefers: {best_aicc}")
        print(f"BIC prefers: {best_bic}")
        if best_aic != best_aicc:
            print(
                "AIC and AICc disagree on the preferred model -- given n_data/k is "
                "low for at least one model here, trust AICc over plain AIC."
            )
        if best_aic != best_bic:
            print(
                "AIC and BIC disagree -- BIC penalizes extra parameters more heavily "
                "(ln(n_data) vs 2), so this usually means the added complexity in the "
                "AIC-preferred model helps a little, but not enough to justify it by "
                "the stricter BIC standard."
            )
        for label in df.index:
            d = df.loc[label, "delta_AICc"]
            if d < 2:
                verdict = "essentially indistinguishable from the best model"
            elif d < 10:
                verdict = "some support against, relative to the best model"
            else:
                verdict = "effectively ruled out relative to the best model"
            print(f"  {label}: delta_AICc={d:.3g} -> {verdict}")

    return df


# ----------------------------------------------------------------------
# Helper: fit_opt / n_params / n_data for one model, given its own setup
# ----------------------------------------------------------------------

def evaluate_model(cost_func, param_ode, calibr_setup, jac_spasity=None):
    """
    Compute the three ingredients compare_models() needs for one already-
    fitted model: cost_opt, n_data, n_params.

    cost_func : callable cost_func(param, calibr_setup, jac_spasity) -> float,
                the same convention used throughout `fusion_core`.
    """
    cost_opt = cost_func(param_ode, calibr_setup, jac_spasity)
    n_data = _likelihood.count_data_points(cost_func, param_ode, calibr_setup, jac_spasity)
    n_params = len(_likelihood.free_param_indices(calibr_setup["param_bnds"]))
    return {"cost_opt": cost_opt, "n_data": n_data, "n_params": n_params}

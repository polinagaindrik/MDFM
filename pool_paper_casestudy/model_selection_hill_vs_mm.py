"""
Model selection: Hill vs Michaelis-Menten (MM) death-term models
====================================================================

Compares the full 3-parameter Hill death term against the 2-parameter
Michaelis-Menten simplification using AIC and BIC, computed under the
same i.i.d.-Gaussian-residual assumption used throughout the profile
likelihood analysis (see profile_likelihood.py).

Unlike the nested likelihood-ratio test (valid only because MM is a
special case of Hill, n=1), AIC/BIC don't require nesting and are
computed independently for each model -- each model gets its own
maximum-likelihood noise estimate sigma_hat^2, rather than sharing one
scale across models. This is the standard approach for comparing
non-nested regression models (Burnham & Anderson).

    AIC = n*ln(RSS/n) + 2*(k+1)
    BIC = n*ln(RSS/n) + (k+1)*ln(n)

RSS/n = cost_opt (your cost_arithmetic_mean already returns the mean,
so RSS = cost_opt * n_data). The "+1" in (k+1) accounts for sigma^2
itself being an estimated parameter, standard convention for Gaussian
least-squares AIC/BIC.

Lower AIC/BIC = preferred model. Delta-AIC/BIC and (for AIC) Akaike
weights are reported for interpretability:
    - Delta < 2   : models are essentially indistinguishable
    - Delta 2-10  : some to strong support for the lower-scoring model
    - Delta > 10  : the higher-scoring model is effectively ruled out

Run directly (python model_selection_hill_vs_mm.py) after editing the
CONFIG block below, or import compare_models() and call it yourself.
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *
from pool_paper_casestudy.do_local_optim import cost
from profile_likelihood import count_data_points, free_param_indices


# ----------------------------------------------------------------------
# AIC / BIC computation
# ----------------------------------------------------------------------

def compute_aic_bic(cost_opt, n_data, n_params):
    """
    cost_opt  : cost_arithmetic_mean at this model's own optimum (RSS/n_data)
    n_data    : number of individual residual terms (from count_data_points)
    n_params  : number of free model parameters (NOT including sigma)

    Returns dict with RSS, neg2logL, k (n_params+1), AIC, BIC.
    """
    RSS = cost_opt * n_data
    if RSS <= 0:
        raise ValueError(f"RSS={RSS} <= 0; can't take log. Check cost_opt/n_data.")

    neg2logL = n_data * np.log(RSS / n_data) + n_data * (np.log(2 * np.pi) + 1)
    k = n_params #+ 1  # +1 for sigma^2, estimated via MLE alongside the model params

    AIC = neg2logL + 2 * k
    BIC = neg2logL + k * np.log(n_data)

    return {
        "cost_opt": cost_opt,
        "RSS": RSS,
        "n_data": n_data,
        "n_params": n_params,
        "k": k,
        "neg2logL": neg2logL,
        "AIC": AIC,
        "BIC": BIC,
    }


def compare_models(model_results, verbose=True):
    """
    model_results : dict {model_label: {"cost_opt":..., "n_data":..., "n_params":...}}
                     (n_data should be the same across models if they were fit
                     to the same dataset -- a mismatch is flagged below)

    Returns a DataFrame with AIC, BIC, delta-AIC, delta-BIC, and Akaike
    weights for each model, sorted by AIC (best first).
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
    df["delta_BIC"] = df["BIC"] - df["BIC"].min()

    # Akaike weights: relative likelihood of each model given the set
    rel_likelihood = np.exp(-0.5 * df["delta_AIC"])
    df["akaike_weight"] = rel_likelihood / rel_likelihood.sum()

    if verbose:
        print("\nModel comparison (sorted by AIC, best first):")
        print(df[["cost_opt", "n_data", "n_params", "AIC", "delta_AIC", "akaike_weight", "BIC", "delta_BIC"]]
              .to_string(float_format=lambda x: f"{x:.6g}"))

        best_aic = df.index[0]
        best_bic = df.sort_values("BIC").index[0]
        print(f"\nAIC prefers: {best_aic}")
        print(f"BIC prefers: {best_bic}")
        if best_aic != best_bic:
            print(
                "AIC and BIC disagree -- BIC penalizes extra parameters more heavily "
                "(ln(n_data) vs 2), so this usually means the added complexity in the "
                "AIC-preferred model helps a little, but not enough to justify it by "
                "the stricter BIC standard."
            )
        for label in df.index:
            d = df.loc[label, "delta_AIC"]
            if d < 2:
                verdict = "essentially indistinguishable from the best model"
            elif d < 10:
                verdict = "some support against, relative to the best model"
            else:
                verdict = "effectively ruled out relative to the best model"
            print(f"  {label}: delta_AIC={d:.3g} -> {verdict}")

    return df


# ----------------------------------------------------------------------
# Helper: fit_opt / n_params / n_data for one model, given its own setup
# ----------------------------------------------------------------------

def evaluate_model(param_ode, calibr_setup, jac_spasity=None):
    """
    Compute the three ingredients compare_models() needs for one already-
    fitted model: cost_opt, n_data, n_params.
    """
    cost_opt = cost(param_ode, calibr_setup, jac_spasity)
    n_data = count_data_points(param_ode, calibr_setup, jac_spasity)
    n_params = len(free_param_indices(calibr_setup["param_bnds"]))
    return {"cost_opt": cost_opt, "n_data": n_data, "n_params": n_params}


if __name__ == "__main__":
    # ================================================================
    # EDIT THESE, THEN RUN THIS FILE DIRECTLY
    # ================================================================
    path2 = "pool_paper_casestudy/out/wo_pH_new/"
    n_cl = 4

    dfs = pd.read_pickle(path2 + "dataframe_poolpaper_all.pkl")
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    data_array = extract_observables_from_df([dfs])

    calibr_base = {
        "output_path": path2,
        "n_cl": n_cl,
        "dfs": [dfs],
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps,
        "data_array": data_array,
    }

    # ---- Hill model (3-parameter death term: omega, K, n) ----
    HILL_JSON = "Result_calibration_5exps_local.json"
    result_hill = fm.output.read_from_json(HILL_JSON, dir=path2)
    param_opt_hill = np.array(result_hill["param_ode"])
    x0_hill = param_opt_hill[: n_cl * len(exps)]
    param_ode_hill = param_opt_hill[n_cl * len(exps):]

    calibr_hill = dict(calibr_base)
    calibr_hill["model"] = ode_model_coculture_wopH
    calibr_hill["x0"] = x0_hill
    calibr_hill["param_bnds"] = tuple(
        [(.2, 1.) for _ in range(3)] +           # mu_opt
        [(0.5, 2.), (1., 8000.), (0.3, 1.5)] +    # omegaT_exp + k_T_inhib + n
        [(8., 9.), (8., 9.), (8., 9.)] +          # N_max_exp
        [(.1, 1.)] +                              # kappa_T
        [(.1, 10)] + [(1., 100.)] +
        [(.1, 10)] + [(1., 100.)] +
        [(.1, 10)] + [(1., 100.)]
    )

    # ---- MM model (2-parameter death term: omega3, K3) ----
    MM_JSON = "Result_calibration_5exps_MM_local.json"
    result_mm = fm.output.read_from_json(MM_JSON, dir=path2)
    param_opt_mm = np.array(result_mm["param_ode"])
    x0_mm = param_opt_mm[: n_cl * len(exps)]
    param_ode_mm = param_opt_mm[n_cl * len(exps):]

    calibr_mm = dict(calibr_base)
    calibr_mm["model"] = ode_model_coculture_wopH_MM
    calibr_mm["x0"] = x0_mm
    calibr_mm["param_bnds"] = tuple(
        [(.33, .39), (.36, .45), (.3, 0.36)] +    # mu_opt
        [(0.46, .6), (0.5, 4)] +                  # omega3, K3
        [(8.05, 8.35), (8.2, 8.4), (8.55, 8.95)] +  # N_max_exp
        [(.45, .95)] +                             # kappa_T
        [(0.4, 0.9)] + [(2., 5.)] +
        [(0.15, 0.6)] + [(3.5, 6.5)] +
        [(0.0, 0.0)] + [(0., 1.5)]
    )
    # ================================================================

    hill_stats = evaluate_model(param_ode_hill, calibr_hill)
    mm_stats = evaluate_model(param_ode_mm, calibr_mm)

    print(f"Hill: cost_opt={hill_stats['cost_opt']:.6g}, n_data={hill_stats['n_data']}, "
          f"n_params={hill_stats['n_params']}")
    print(f"MM:   cost_opt={mm_stats['cost_opt']:.6g}, n_data={mm_stats['n_data']}, "
          f"n_params={mm_stats['n_params']}")

    df_comparison = compare_models({
        "Hill": hill_stats,
        "MM": mm_stats,
    })

    df_comparison.to_csv(path2 + "model_selection_hill_vs_mm.csv")
    print(f"\nSaved comparison table to {path2}model_selection_hill_vs_mm.csv")

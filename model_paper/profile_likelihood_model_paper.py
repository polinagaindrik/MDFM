"""
Profile likelihood for model_paper (MDFM repo)
==============================================

Counterpart of ``pool_paper_casestudy/profile_likelihood.py``. The engine
(fusion_model/parameter_estimation/profile_likelihood.py) has the same API as
``pool_paper_casestudy/fusion_core/likelihood.py``; this file only supplies the
case-specific plumbing: it injects ``cost`` (= fm.pest.cost_withS) and
re-exports every function under the same name/signature as the pool_paper
wrappers, so calls can be copied between the two projects.

Parameter vector layout (cost_withS, model fusion_model2):
    [ x0 (n_cl per experiment, log10) | lambda_1 (n_cl) | lambda_exp (n_cl) |
      alpha_0 (n_cl) | alpha_exp (n_cl) | N_1 | N_exp | k_ij (n_cl x n_cl) |
      S (n_media x n_cl) ]
Fixed entries (k_ii = 0, predefined S) have lower == upper bound and are
neither profiled nor re-optimized.

Run from the repository root:
    python model_paper/profile_likelihood_model_paper.py
Re-plot afterwards (without recomputing):
    python model_paper/plot_profile_from_csv.py
"""
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl

import numpy as np
import pandas as pd

cost = fm.pest.cost_withS  # cost(param, calibr_setup, jac_spasity)


# ----------------------------------------------------------------------
# Thin case-study wrappers (same names/signatures as pool_paper_casestudy/profile_likelihood.py)
# ----------------------------------------------------------------------
free_param_indices = pl.free_param_indices
confidence_interval_from_profile = pl.confidence_interval_from_profile
plot_profile_likelihood = pl.plot_profile_likelihood


def count_data_points(param, calibr_setup, jac_spasity=None):
    """See `profile_likelihood.count_data_points` (this case study's `cost` is used)."""
    return pl.count_data_points(cost, param, calibr_setup, jac_spasity)


def estimate_profile_scale(param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=None):
    """See `profile_likelihood.estimate_profile_scale` (this case study's `cost` is used)."""
    return pl.estimate_profile_scale(cost, param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=jac_spasity)


def profile_likelihood_for_param(param_opt, param_index, calibr_setup, *args, **kwargs):
    """See `profile_likelihood.profile_likelihood_for_param` (this case study's `cost` is used)."""
    return pl.profile_likelihood_for_param(cost, param_opt, param_index, calibr_setup, *args, **kwargs)


def run_profile_likelihood_all(param_opt, calibr_setup, *args, **kwargs):
    """See `profile_likelihood.run_profile_likelihood_all` (this case study's `cost` is used)."""
    return pl.run_profile_likelihood_all(cost, param_opt, calibr_setup, *args, **kwargs)


# ----------------------------------------------------------------------
# Loading the model_paper calibration
# ----------------------------------------------------------------------

def load_calibration(path, add_name):
    """Data frames, estimated parameters and (if present) true in-silico parameters."""
    dfs = []
    for k in ["mibi", "maldi", "ngs"]:
        f_pre = os.path.join(path, f"dataframe_{k}{add_name}_preprocessed.pkl")
        f_raw = os.path.join(path, f"dataframe_{k}{add_name}.pkl")
        dfs.append(pd.read_pickle(f_pre if os.path.exists(f_pre) else f_raw))
    result = fm.output.read_from_json(f"Result_calibration{add_name}.json", dir=path)
    true_params = None
    if os.path.exists(os.path.join(path, "Result_temp_together_real.json")):
        tr = fm.output.read_from_json("Result_temp_together_real.json", dir=path)
        true_params = np.concatenate([np.ravel(tr["param_ode"]), np.ravel(tr["s_x"])])
    return dfs, result, true_params


def s_x_predefined_from_maldi(df_maldi, media):
    """Same rule as dtf.make_df_maldi_ngs_compatible: S fixed to 0 for bacteria
    never seen on a medium, free (NaN) otherwise."""
    s = np.full((len(media), np.shape(df_maldi)[0]), np.nan)
    for i, med in enumerate(media):
        zero = (df_maldi.filter(like=med).T.sum() == 0).to_numpy()
        s[i, zero] = 0.
    return s


def exp_temps_from_columns(df):
    """{exp: temperature} parsed from column names like 'V01_M1_02C_00_MRS-mibi'."""
    temps = {}
    for c in df.columns:
        parts = c.split("_")
        temps.setdefault(parts[0], float(parts[2].rstrip("C")))
    return temps


def make_param_names(exps, bact, media):
    n_cl = len(bact)
    b = [s.split("_")[-1] for s in bact]  # 'Bacteria_03' -> '03'
    names = [rf"$x_{{0,{bi}}}^{{{e}}}$" for e in exps for bi in b]
    names += [rf"$\lambda_{{1,{bi}}}$" for bi in b]
    names += [rf"$\lambda_{{exp,{bi}}}$" for bi in b]
    names += [rf"$\alpha_{{0,{bi}}}$" for bi in b]
    names += [rf"$\alpha_{{exp,{bi}}}$" for bi in b]
    names += [r"$N_1$", r"$N_{exp}$"]
    names += [rf"$k_{{{b[i]},{b[j]}}}$" for i in range(n_cl) for j in range(n_cl)]
    names += [rf"$S_{{{m},{bi}}}$" for m in media for bi in b]
    return names


def build_calibr_setup(path, add_name, n_cl, n_media, objective="neg2logL"):
    """calibr_setup (incl. param_bnds and data_array) for a saved calibration.
    objective: 'neg2logL' (GaussianNeg2LogLik, scale=1) or 'calibration'
    (cost_sum_and_geometric_mean, as used for calibration)."""
    dfs, result, true_params = load_calibration(path, add_name)
    df_mibi, df_maldi, df_ngs = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_mibi.columns])))
    media = sorted(list(set([s.split("_")[-1].split("-")[0] for s in df_maldi.columns])))
    bact = list(df_maldi.index)
    assert len(bact) == n_cl and len(media) == n_media, (bact, media)
    calibr_presetup = {
        "model": fm.mdl.fusion_model2,
        "T_x": result["T_x"],
        "workers": 1,
        "output_path": path,
        "n_cl": n_cl,
        "n_media": n_media,
        "dfs": dfs,
        "aggregation_func": fm.pest.cost_sum_and_geometric_mean,
        "exps": exps,
        "exp_temps": exp_temps_from_columns(df_mibi),
        "media": media,
    }
    calibr_setup = fm.pest.define_calibr_setup_insilico(
        calibr_presetup, inhib=True, s_x_predefined=s_x_predefined_from_maldi(df_maldi, media), s_x=None)
    calibr_setup["data_array"] = fm.dtf.extract_observables_from_df(dfs)
    if objective == "neg2logL":
        calibr_setup["aggregation_func"] = pl.make_gaussian_neg2loglik(calibr_setup["data_array"])
    param_opt = np.array(result["param_ode"], dtype=float)
    param_names = make_param_names(exps, bact, media)
    assert len(param_opt) == len(calibr_setup["param_bnds"]) == len(param_names)
    return calibr_setup, param_opt, param_names, true_params


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Settings
    # --------------------------------------------------------------
    n_cl = 4
    n_media = 2
    relnoise = 0.1
    add_name = f"_{int(n_cl)}dim_{int(n_media)}media"
    path2 = f"model_paper/out/model_complexity/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/"
    out_path = path2 + "profile_likelihood/"

    PROFILE = "ode+S"       # 'ode', 'ode+S' or 'all' (incl. x0)
    FIX_X0 = False          # True: x0 held at the estimate (faster, profiles conditional on x0)
    OBJECTIVE = "neg2logL"  # 'neg2logL' (scale = 1) or 'calibration' (scale='auto' as in pool_paper)

    os.makedirs(out_path, exist_ok=True)
    calibr_setup, param_opt, ode_param_names, true_params = build_calibr_setup(
        path2, add_name, n_cl, n_media, objective=OBJECTIVE)

    n_exps = len(calibr_setup["exps"])
    n_x0 = n_cl * n_exps
    n_ode = 4 * n_cl + 2 + n_cl * n_cl
    idx_x0 = list(range(n_x0))
    idx_ode = list(range(n_x0, n_x0 + n_ode))
    idx_S = list(range(n_x0 + n_ode, len(param_opt)))
    profile_indices = {"ode": idx_ode, "ode+S": idx_ode + idx_S, "all": None}[PROFILE]
    tag = f"{add_name}_{PROFILE.replace('+', '')}_{OBJECTIVE}" + ("_fixx0" if FIX_X0 else "")

    df, cis = run_profile_likelihood_all(
        param_opt, calibr_setup,
        span=1., n_points=30, method="local",
        n_jobs=20,              # <-- parallelize across (parameter, direction) walks
        per_point_workers=1,    # <-- irrelevant for method="local", leave at 1
        out_csv=out_path + f"profile_likelihood_results{tag}.csv",
        plot_path=out_path + f"profile_likelihood{tag}.png",
        param_names=ode_param_names,
        n_restarts=1, jitter_frac=0.05,
        # optional extras (model_paper):
        profile_indices=profile_indices,
        fixed_indices=idx_x0 if FIX_X0 else (),
        refine=True,            # polish the optimum with the profiling objective first
        stop_factor=1.5,        # stop a walk once Delta > 1.5 * threshold (None: whole grid)
        true_params=true_params,
    )

    '''
    # Re-run the single-parameter profile for one parameter, e.g. N_1
    IDX = ode_param_names.index(r"$N_1$")
    grid, profile_cost, profile_params = profile_likelihood_for_param(
        param_opt, IDX, calibr_setup,
        span=0.2, n_points=15, method="local",
        n_jobs=2, n_restarts=1, jitter_frac=0.05,
    )
    cost_opt = cost(param_opt, calibr_setup, None)
    scale, n_data, sigma_hat2 = estimate_profile_scale(
        param_opt, calibr_setup, cost_opt,
        n_free_params=len(free_param_indices(calibr_setup["param_bnds"])),
    )
    print(confidence_interval_from_profile(grid, profile_cost, cost_opt, scale=scale))
    '''

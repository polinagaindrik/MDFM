"""
Profile likelihood / identifiability analysis for the model_paper calibration
=============================================================================

Loads a saved calibration (Result_calibration<add_name>.json + the data frames
written by main_paper_calibr.py), rebuilds the calibration setup and runs a
profile likelihood with local (L-BFGS-B) re-optimization, using the engine in
fusion_model/parameter_estimation/profile_likelihood.py
(same approach as pool_paper_casestudy/profile_likelihood.py).

Parameter vector layout (cost_withS, model fusion_model2):
    [ x0 (n_cl per experiment, log10) | lambda_1 (n_cl) | lambda_exp (n_cl) |
      alpha_0 (n_cl) | alpha_exp (n_cl) | N_1 | N_exp | k_ij (n_cl x n_cl) |
      S (n_media x n_cl) ]
Fixed entries (k_ii = 0, predefined S) have lower == upper bound and are
neither profiled nor re-optimized.

Run from the repository root:
    python model_paper/profile_likelihood_model_paper.py
"""
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl

import numpy as np
import pandas as pd


def load_calibration(path, add_name):
    """Data frames, estimated parameters and (if present) true in-silico parameters."""
    dfs = []
    for k in ["mibi", "maldi", "ngs"]:
        f_pre = os.path.join(path, f"dataframe_{k}{add_name}_preprocessed.pkl")
        f_raw = os.path.join(path, f"dataframe_{k}{add_name}.pkl")
        dfs.append(pd.read_pickle(f_pre if os.path.exists(f_pre) else f_raw))
    result = fm.output.read_from_json(f"Result_calibration{add_name}.json", dir=path)
    true_file = os.path.join(path, "Result_temp_together_real.json")
    true_params = None
    if os.path.exists(true_file):
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


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    n_cl = 4
    n_media = 2
    add_name = f"_{n_cl}dim_{n_media}media"
    path = f"model_paper/out/{n_cl}_dim/calibration/"
    out_path = f"model_paper/out/{n_cl}_dim/profile_likelihood/"

    # What to profile: 'ode' (growth/interaction parameters), 'ode+S', or 'all' (incl. x0)
    PROFILE = "ode+S"
    # True: initial values x0 are held at the estimate (much faster, but the
    # profiles are then conditional on x0 and therefore somewhat too narrow).
    # False: x0 are re-optimized as nuisance parameters at every profile point.
    FIX_X0 = False

    # Objective: 'neg2logL' -> Gaussian -2 log L with one noise variance per data
    # type (chi2 threshold applies directly, recommended); 'calibration' -> the
    # cost used for calibration (cost_sum_and_geometric_mean) with a manual SCALE.
    OBJECTIVE = "neg2logL"
    SCALE = 1.0  # only used for OBJECTIVE='calibration'

    N_JOBS = 16  # one worker per (parameter, direction) walk
    # ================================================================

    os.makedirs(out_path, exist_ok=True)
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
        "output_path": out_path,
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

    if OBJECTIVE == "neg2logL":
        calibr_setup["aggregation_func"] = pl.make_gaussian_neg2loglik(calibr_setup["data_array"])
        scale = 1.0
    else:
        scale = SCALE

    param_opt = np.array(result["param_ode"], dtype=float)
    param_names = make_param_names(exps, bact, media)
    assert len(param_opt) == len(calibr_setup["param_bnds"]) == len(param_names)

    # index blocks
    n_x0 = n_cl * len(exps)
    n_ode = 4 * n_cl + 2 + n_cl * n_cl
    idx_x0 = list(range(n_x0))
    idx_ode = list(range(n_x0, n_x0 + n_ode))
    idx_S = list(range(n_x0 + n_ode, len(param_opt)))
    profile_indices = {"ode": idx_ode, "ode+S": idx_ode + idx_S, "all": None}[PROFILE]
    fixed_indices = idx_x0 if FIX_X0 else ()

    tag = f"{add_name}_{PROFILE.replace('+', '')}_{OBJECTIVE}" + ("_fixx0" if FIX_X0 else "")
    df, ci, param_ref, cost_ref = pl.run_profile_likelihood(
        fm.pest.cost_withS, param_opt, calibr_setup,
        profile_indices=profile_indices,
        fixed_indices=fixed_indices,
        refine=True,
        scale=scale,
        confidence_level=0.95,
        init_step_frac=0.01, max_step_frac=0.1, max_points=30, stop_factor=1.5,
        anchor_start=True, n_jitter=0,
        n_jobs=N_JOBS,
        param_names=param_names,
        out_csv=os.path.join(out_path, f"profile_likelihood{tag}.csv"),
        plot_path=os.path.join(out_path, f"profile_likelihood{tag}.png"),
        true_params=true_params,
    )

    # To re-plot later without recomputing:
    # pl.plot_profile_likelihood_from_csv(os.path.join(out_path, f"profile_likelihood{tag}.csv"),
    #                                     true_params=true_params, save_path=...)

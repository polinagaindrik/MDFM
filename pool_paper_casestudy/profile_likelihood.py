"""
Profile likelihood for pool_paper_casestudy (MDFM repo)
=========================================================

The actual profile-likelihood engine (grid construction, fix-one-parameter /
re-optimize-the-rest sweeps, multiprocessing across grid points, chi2-based
confidence intervals and plotting) doesn't depend on anything specific to
this case study -- it only needs a `cost_func(param, calibr_setup,
jac_spasity)` callable, the same convention used throughout this codebase
(see `fusion_core.pest.calculate_model_params`). That generic engine now
lives in `pool_paper_casestudy.fusion_core.likelihood`, alongside the other
commonly-used source code in `fusion_core` (it plays the same role there as
`fusion_model/parameter_estimation/likelihood_functions.py` does for the
main `fusion_model` package, which uses a different, incompatible
`ll_func(param, dfs, model, P_matrix, s_x)` signature and so can't be reused
as-is for this case study).

This file now only holds the case-study-specific plumbing: it supplies
`cost` (imported from `pool_paper_casestudy/do_local_optim.py`) to
`fusion_core.likelihood`, and thinly re-exports every function under its
original name/signature so existing call sites (here and in
`compare_models.py`, `model_selection_hill_vs_mm.py`,
`model_selection_pH_vs_noPH.py`) keep working unchanged.

CAVEAT ON THE CHI2 THRESHOLD
-----------------------------
`cost_arithmetic_mean` (the aggregation function used here) returns a *mean
squared residual*, not `-2*logL`. The chi2-based confidence interval below is
only statistically exact if `cost` is proportional to `-2*logL` (e.g.
`n * MSE / sigma^2` under i.i.d. Gaussian noise). Treat `confidence_interval_from_profile`
as a convenience utility -- calibrate `scale` (or pass your own threshold) if you
need a rigorous interval. This mirrors what `likelihood_functions.py` already
does elsewhere in the repo, so it is consistent with how the rest of the codebase
reports these intervals, but it is an approximation, not a guarantee.

CAVEAT ON MULTIPROCESSING
---------------------------
`method="global"` already parallelizes *within* each grid point via
`differential_evolution(..., workers=...)`. Also parallelizing *across* grid
points (n_jobs > 1) on top of that oversubscribes your CPU cores. If you use
`method="global"`, keep `n_jobs=1` (default) and control parallelism only
through `per_point_workers`, or explicitly divide your cores between the two
levels (n_jobs * per_point_workers <= number of cores).
`method="local"` (L-BFGS-B) is single-threaded per point, so `n_jobs > 1` is
the recommended way to parallelize it.

Drop this file into `pool_paper_casestudy/` (same level as `do_local_optim.py`)
so the relative imports resolve, or adjust the `sys.path` / import lines below.
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from pool_paper_casestudy import fusion_core as fm
from pool_paper_casestudy.fusion_core.dtf import extract_observables_from_df
from pool_paper_casestudy.fusion_core.mdl import ode_model_coculture_wopH_MM
from pool_paper_casestudy.local_optimization import cost  # your cost(param, calibr_setup, jac_spasity)


# ----------------------------------------------------------------------
# Thin case-study wrappers around fusion_core.likelihood
# ----------------------------------------------------------------------
# Every wrapper below just injects this case study's `cost` function as the
# `cost_func` argument of the generic engine in `fusion_core.likelihood`, and
# keeps the exact original name/signature so nothing else in the repo needs
# to change.

free_param_indices = fm.likelihood.free_param_indices
confidence_interval_from_profile = fm.likelihood.confidence_interval_from_profile
plot_profile_likelihood = fm.likelihood.plot_profile_likelihood


def count_data_points(param, calibr_setup, jac_spasity=None):
    """See `fusion_core.likelihood.count_data_points` (this case study's `cost` is used)."""
    return fm.likelihood.count_data_points(cost, param, calibr_setup, jac_spasity)


def estimate_profile_scale(param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=None):
    """See `fusion_core.likelihood.estimate_profile_scale` (this case study's `cost` is used)."""
    return fm.likelihood.estimate_profile_scale(
        cost, param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=jac_spasity
    )


def profile_likelihood_for_param(param_opt, param_index, calibr_setup, *args, **kwargs):
    """See `fusion_core.likelihood.profile_likelihood_for_param` (this case study's `cost` is used)."""
    return fm.likelihood.profile_likelihood_for_param(cost, param_opt, param_index, calibr_setup, *args, **kwargs)


def run_profile_likelihood_all(param_opt, calibr_setup, *args, **kwargs):
    """See `fusion_core.likelihood.run_profile_likelihood_all` (this case study's `cost` is used)."""
    return fm.likelihood.run_profile_likelihood_all(cost, param_opt, calibr_setup, *args, **kwargs)


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Example usage -- adapt to how you actually built calibr_setup
    # and obtained param_opt in your run of do_local_optim.py /
    # case_study_poolpaper2.py.
    # --------------------------------------------------------------
    path2 = "pool_paper_casestudy/out/wo_pH/"
    n_cl = 4
    add_name = '_MM'

    result = fm.output.read_from_json(f"Result_calibration_5exps{add_name}_local.json", dir=path2)
    param_opt = np.array(result["param_ode"])
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    data_array = extract_observables_from_df([dfs])
    x0_vals = param_opt[:n_cl*len(exps)]
    param_ode = param_opt[n_cl*len(exps):]
    model = ode_model_coculture_wopH_MM
    calibr_presetup = {
            "model": model,
            "output_path": path2,
            "n_cl": n_cl,
            "dfs": [dfs],
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps,
            'data_array': data_array,
            'x0': x0_vals
    }
    param_ode_bnds = tuple(
            [(.34, .38), (.38, .44), (.32, .355)] + # mu_opt
            #[(0.9, 1.2), (700., 14000.), (0.25, 0.5)] + # omegaT_exp + ki_T_inhib + n 
            [(0.46, .6), (0.6, 3)] + 
            [(8.13, 8.34), (8.22, 8.4), (8.55, 8.95)]  + # N_max_exp
            [(.45, .85)] + # kappa_T
            [(0.45, 0.85)] + [(2.1, 4.5)] +   # kappa_LA ls23K
            [(0.15, 0.45)] + [(3.5, 6)] +   # kappa_LA lsCTC494
            [(0., 0.0)] + [(0.2, 1.2)]     # kappa_LA lm
        )
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = param_ode_bnds

    ode_param_names = [
        r"$\mu_{Ls23K}$", r"$\mu_{LsCTC494}$", r"$\mu_{Lm}$",
        r"$\omega_T^{Lm}$", r"$K \cdot 10^{-2}$",
        r"$\log_{10}{N^{Ls23K}_{t}}$", r"$\log_{10}{N^{LsCTC494}_{t}}$", r"$\log_{10}{N^{Lm}_{t}}$",
        r"$\kappa_{T} \cdot 10^{-5}$",
        r"$\kappa_{LA}^{Ls23K} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{Ls23K} \cdot 10^{-9}$",
        r"$\kappa_{LA}^{LsCTC494} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{LsCTC494} \cdot 10^{-9}$",
        r"$\kappa_{LA}^{Lm} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{Lm} \cdot 10^{-9}$",
    ]
    
    df, cis = run_profile_likelihood_all(
        param_ode, calibr_setup,
        span=2., n_points=20, method="local",
        n_jobs=20,              # <-- parallelize across grid points
        per_point_workers=1,    # <-- irrelevant for method="local", leave at 1
        out_csv=path2+f"profile_likelihood_results{add_name}.csv",
        plot_path=path2+f"profile_likelihood{add_name}.png",
        param_names=ode_param_names,
        n_restarts=5, jitter_frac=0.05
    )

    '''

    # Re-run the single-parameter profile for omega
    OMEGA_INDEX = 3

    grid, profile_cost, profile_params = profile_likelihood_for_param(
        param_ode, OMEGA_INDEX, calibr_setup,
        span=3.0, n_points=15, method="local",
        n_jobs=20, n_restarts=6, jitter_frac=0.05,
    )

    cost_opt = cost(param_ode, calibr_setup, None)

    scale, n_data, sigma_hat2 = estimate_profile_scale(
        param_ode, calibr_setup, cost_opt,
        n_free_params=len(free_param_indices(calibr_setup["param_bnds"])),
    )
    print(f"scale={scale:.6g}, n_data={n_data}, sigma_hat2={sigma_hat2:.6g}")

    # --- insert here: merge new omega rows into the existing MM results CSV ---
    OUT_CSV = path2 + f"profile_likelihood_results{add_name}.csv"

    new_rows = []
    for g, c, full_p in zip(grid, profile_cost, profile_params):
        row = {"param_index": OMEGA_INDEX, "param_value": g, "cost": c}
        for j, pv in enumerate(full_p):
            row[ode_param_names[j]] = pv
        new_rows.append(row)
    df_new = pd.DataFrame(new_rows)

    if os.path.exists(OUT_CSV):
        df_existing = pd.read_csv(OUT_CSV)
        df_existing = df_existing[df_existing["param_index"] != OMEGA_INDEX]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values(["param_index", "param_value"]).reset_index(drop=True)
    df_combined.to_csv(OUT_CSV, index=False)
    print(f"Updated {OUT_CSV}: {len(df_new)} rows for param_index={OMEGA_INDEX}")
    # --- end insert ---

    df_k = pd.DataFrame({"param_index": OMEGA_INDEX, "param_value": grid, "cost": profile_cost})

    fig = plot_profile_likelihood(
        df_k, param_ode, cost_opt,
        ci_results={OMEGA_INDEX: confidence_interval_from_profile(grid, profile_cost, cost_opt, scale=scale)},
        scale=scale,
        param_names={OMEGA_INDEX: ode_param_names[OMEGA_INDEX]},
        save_path=path2 + "profile_omega_only.png",
    )
    ''' 
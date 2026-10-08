"""
Model selection: Hill vs Michaelis-Menten (MM) death-term models
====================================================================

The AIC/BIC comparison engine (`compute_aic_bic`, `compare_models`) is
generic -- it doesn't depend on anything specific to the Hill-vs-MM
comparison or to this case study -- so it now lives in
`pool_paper_casestudy.fusion_core.model_selection` alongside the other
common source code in `fusion_core`. `evaluate_model` only needed this
case study's `cost` (from `do_local_optim.py`) supplied as its `cost_func`
argument.

This file keeps just that case-study wiring and the worked CONFIG/__main__
example below. See `fusion_core.model_selection` for the AIC/BIC math and
its docstring.

Run directly (python model_selection_hill_vs_mm.py) after editing the
CONFIG block below, or import compare_models() and call it yourself.
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from pool_paper_casestudy import fusion_core as fm
from pool_paper_casestudy.fusion_core.dtf import extract_observables_from_df
from pool_paper_casestudy.fusion_core.mdl import ode_model_coculture_wopH, ode_model_coculture_wopH_MM
from pool_paper_casestudy.local_optimization import cost
from pool_paper_casestudy.profile_likelihood import count_data_points, free_param_indices


# ----------------------------------------------------------------------
# Thin case-study wrappers around fusion_core.model_selection
# ----------------------------------------------------------------------
compute_aic_bic = fm.model_selection.compute_aic_bic
compare_models = fm.model_selection.compare_models


def evaluate_model(param_ode, calibr_setup, jac_spasity=None):
    """See `fusion_core.model_selection.evaluate_model` (this case study's `cost` is used)."""
    return fm.model_selection.evaluate_model(cost, param_ode, calibr_setup, jac_spasity)


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

    hill_full = compute_aic_bic(hill_stats["cost_opt"], hill_stats["n_data"], hill_stats["n_params"])
    mm_full = compute_aic_bic(mm_stats["cost_opt"], mm_stats["n_data"], mm_stats["n_params"])

    print(f"Hill: -2logL={hill_full['neg2logL']:.6g}, sigma^2={hill_full['sigma_hat2']:.6g}, "
          f"n_data={hill_full['n_data']}, n_params={hill_full['n_params']}, k={hill_full['k']}")
    print(f"MM:   -2logL={mm_full['neg2logL']:.6g}, sigma^2={mm_full['sigma_hat2']:.6g}, "
          f"n_data={mm_full['n_data']}, n_params={mm_full['n_params']}, k={mm_full['k']}")

    df_comparison = compare_models({
        "Hill": hill_stats,
        "MM": mm_stats,
    })

    df_comparison.to_csv(path2 + "model_selection_hill_vs_mm.csv")
    print(f"\nSaved comparison table to {path2}model_selection_hill_vs_mm.csv")

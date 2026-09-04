"""
Model selection: pH-dependent vs pH-independent (wo_pH) models
====================================================================

Compares the full pH-dependent growth model (mu(pH) via pH_min/opt/max
triplets per species) against the simplified model that drops pH
dependence entirely, using AIC/BIC -- same approach as
model_selection_hill_vs_mm.py, reused directly from that file.

These two models are not simply nested by fixing one parameter (dropping
pH dependence removes 9 parameters at once -- 3 species x pH_min/opt/max),
so a standard 1-dof nested likelihood-ratio test doesn't directly apply.
AIC/BIC don't require nesting and instead ask: does the added pH
mechanism explain enough extra variance in this dataset to be worth its
9 extra parameters?

    AIC = n*ln(RSS/n) + 2*(k+1)
    BIC = n*ln(RSS/n) + (k+1)*ln(n)

IMPORTANT: both models must be evaluated on the EXACT same data (same
experiments, same dfs/data_array) for this comparison to be valid --
compare_models() warns if n_data differs between the two runs.

Run directly (python model_selection_pH_vs_noPH.py) after editing the
CONFIG block below -- in particular double-check the pH model's function
name, bounds, and result JSON filename/path against what you actually
saved; the values below are reconstructed from context earlier in this
project and may not exactly match your saved run.
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

# reuse the AIC/BIC machinery instead of duplicating it
from model_selection_hill_vs_mm import compute_aic_bic, compare_models, evaluate_model


if __name__ == "__main__":
    # ================================================================
    # EDIT THESE, THEN RUN THIS FILE DIRECTLY
    # ================================================================
    n_cl = 4

    # ---- pH-independent model (wo_pH) ----
    path_nopH = "pool_paper_casestudy/out/wo_pH_new/"
    NOPH_JSON = "Result_calibration_5exps_local.json"

    dfs_nopH = pd.read_pickle(path_nopH + "dataframe_poolpaper_all.pkl")
    exps_nopH = sorted(list(set([s.split("_")[0] for s in dfs_nopH.columns])))
    data_array_nopH = extract_observables_from_df([dfs_nopH])

    result_nopH = fm.output.read_from_json(NOPH_JSON, dir=path_nopH)
    param_opt_nopH = np.array(result_nopH["param_ode"])
    x0_nopH = param_opt_nopH[: n_cl * len(exps_nopH)]
    param_ode_nopH = param_opt_nopH[n_cl * len(exps_nopH):]

    calibr_nopH = {
        "output_path": path_nopH,
        "n_cl": n_cl,
        "dfs": [dfs_nopH],
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps_nopH,
        "data_array": data_array_nopH,
        "x0": x0_nopH,
        "model": ode_model_coculture_wopH,
        "param_bnds": tuple(
            [(.2, 1.) for _ in range(3)] +           # mu_opt
            [(0.5, 2.), (1., 8000.), (0.3, 1.5)] +    # omegaT_exp + k_T_inhib + n
            [(8., 9.), (8., 9.), (8., 9.)] +          # N_max_exp
            [(.1, 1.)] +                              # kappa_T
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)]
        ),
    }

    # ---- pH-dependent model ----
    # NOTE: verify path, JSON filename, model function name, and bounds
    # below against your actual saved with-pH calibration run -- these
    # are reconstructed from earlier context and may need adjustment.
    path_pH = "pool_paper_casestudy/out/lininter_final/"
    PH_JSON = "Result_calibration_5exps_local.json"
    PH_MODEL_FUNC = ode_model_coculture3  # <-- confirm this matches what was actually fit

    dfs_pH = pd.read_pickle(path_pH + "dataframe_poolpaper_all.pkl")
    exps_pH = sorted(list(set([s.split("_")[0] for s in dfs_pH.columns])))
    data_array_pH = extract_observables_from_df([dfs_pH])

    result_pH = fm.output.read_from_json(PH_JSON, dir=path_pH)
    param_opt_pH = np.array(result_pH["param_ode"])
    x0_pH = param_opt_pH[: n_cl * len(exps_pH)]
    param_ode_pH = param_opt_pH[n_cl * len(exps_pH):]

    calibr_pH = {
        "output_path": path_pH,
        "n_cl": n_cl,
        "dfs": [dfs_pH],
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps_pH,
        "data_array": data_array_pH,
        "x0": x0_pH,
        "model": PH_MODEL_FUNC,
        "param_bnds": tuple(
            [(.2, 1.) for _ in range(3)] +            # mu_opt
            [(1., 3.5), (6., 8.), (9., 14.)] +          # pH_ls23K_min, opt, max
            [(1., 3.5), (6., 8.), (9., 14.)] +          # pH_lsCTC494_min, opt, max
            [(1., 4.), (6., 8.), (9., 14.)] +           # pH_lm_min, opt, max
            [(0.5, 2.), (1., 8000.), (0.3, 1.5)] +       # omegaT_exp + k_T_inhib + n
            [(8., 9.), (8., 9.), (8., 9.)] +            # N_max_exp
            [(.1, 1.)] +                                 # kappa_T
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)]
        ),
    }
    # ================================================================

    nopH_stats = evaluate_model(param_ode_nopH, calibr_nopH)
    pH_stats = evaluate_model(param_ode_pH, calibr_pH)

    print(f"wo_pH: cost_opt={nopH_stats['cost_opt']:.6g}, n_data={nopH_stats['n_data']}, "
          f"n_params={nopH_stats['n_params']}")
    print(f"pH:    cost_opt={pH_stats['cost_opt']:.6g}, n_data={pH_stats['n_data']}, "
          f"n_params={pH_stats['n_params']}")

    df_comparison = compare_models({
        "wo_pH": nopH_stats,
        "pH": pH_stats,
    })

    out_csv = path_nopH + "model_selection_pH_vs_noPH.csv"
    df_comparison.to_csv(out_csv)
    print(f"\nSaved comparison table to {out_csv}")

    print(
        "\nInterpretation note: dropping pH dependence removes 9 parameters at once "
        "(3 pH_min/opt/max triplets). A result favoring wo_pH means the pH-response "
        "shape isn't earning its added complexity given this dataset -- which could "
        "mean either (a) pH genuinely doesn't matter much for growth in these "
        "experiments' pH range, or (b) the experiments don't span enough of the pH "
        "range to identify the pH_min/opt/max parameters well (worth checking with "
        "profile likelihood on the pH model specifically, the same way K_inhib's "
        "identifiability was checked earlier)."
    )

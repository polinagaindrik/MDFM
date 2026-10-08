import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from pool_paper_casestudy import fusion_core as fm
from pool_paper_casestudy.fusion_core.dtf import extract_observables_from_df
from pool_paper_casestudy.fusion_core.mdl import ode_model_coculture_wopH_MM, ode_model_coculture_withpH_MM
from pool_paper_casestudy.local_optimization import cost
from pool_paper_casestudy.fusion_core.model_selection import compare_models, evaluate_model

def evaluate_model(param_ode, calibr_setup, jac_spasity=None):
    """See `fusion_core.model_selection.evaluate_model` (this case study's `cost` is used)."""
    return fm.model_selection.evaluate_model(cost, param_ode, calibr_setup, jac_spasity)


if __name__ == "__main__":
    n_cl = 4

    # ---- pH-independent model (wo_pH) ----
    path_nopH = "pool_paper_casestudy/out/wo_pH/"
    NOPH_JSON = "Result_calibration_5exps_MM_local.json"

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
        "model": ode_model_coculture_wopH_MM,
        "param_bnds": tuple(
            [(.2, 1.) for _ in range(3)] +           # mu_opt
            [(0.05, 3.0), (1, 1000)] +              # omegaT_exp + K
            [(8., 9.), (8., 9.), (8., 9.)] +          # N_max_exp
            [(.1, 1.)] +                              # kappa_T
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)] +
            [(0., 0.)] + [(1., 100.)]
        ),
    }

    #  pH-dependent model 
    path_pH = 'pool_paper_casestudy/out/lininter/lininter_all/'
    PH_JSON = "Result_calibration_5exps_MM_local.json"
    PH_MODEL_FUNC = ode_model_coculture_withpH_MM 

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
            [(0.05, 3.0), (1, 1000)] +                  # omegaT_exp + K
            [(8., 9.), (8., 9.), (8., 9.)] +            # N_max_exp
            [(.1, 1.)] +                                 # kappa_T
            [(.1, 10)] + [(1., 100.)] +
            [(.1, 10)] + [(1., 100.)] +
            [(0., 0.)] + [(1., 100.)]
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
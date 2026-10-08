"""
Local refinement of an existing calibration result
==================================================

Runs the L-BFGS-B step (fm.pest.local_optimization) on a calibration that was
already obtained with the global optimizer, without repeating the global search.
New calibrations from main_paper_calibr.py already include this step.

Reads   <path>/Result_calibration<add_name>.json  (+ data frames)
Writes  <path>/Result_calibration<add_name>_global.json  (copy of the input, kept once)
        <path>/Result_calibration<add_name>.json         (refined optimum)
        <path>/optimization_history_local.csv

Run from the repository root:
    python model_paper/main_paper_local_refine.py
"""
import os
import sys
import shutil
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from profile_likelihood_model_paper import load_calibration, s_x_predefined_from_maldi, exp_temps_from_columns

import numpy as np
import time

if __name__ == "__main__":
    n_cl = 4
    n_media = 2
    add_name = f"_{n_cl}dim_{n_media}media"
    path = f"model_paper/out/{n_cl}_dim/calibration/"
    MAXITER = 1000
    N_RESTARTS = 4  # start 0 = global result, others = random perturbations of it
    JITTER_FRAC = 0.02  # size of the perturbations (fraction of the bound width)
    N_JOBS = 4  # starts run in parallel processes (-1 = all cores)

    dfs, result, _ = load_calibration(path, add_name)
    df_mibi, df_maldi, df_ngs = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_mibi.columns])))
    media = sorted(list(set([s.split("_")[-1].split("-")[0] for s in df_maldi.columns])))

    calibr_presetup = {
        "model": fm.mdl.fusion_model2,
        "T_x": result["T_x"],
        "workers": 1,
        "output_path": path,
        "n_cl": n_cl,
        "n_media": n_media,
        "dfs": dfs,
        "aggregation_func": fm.pest.cost_sum_and_geometric_mean,  # same objective as the calibration
        "exps": exps,
        "exp_temps": exp_temps_from_columns(df_mibi),
        "media": media,
    }
    calibr_setup = fm.pest.define_calibr_setup_insilico(
        calibr_presetup, inhib=True, s_x_predefined=s_x_predefined_from_maldi(df_maldi, media), s_x=None)
    calibr_setup["data_array"] = fm.dtf.extract_observables_from_df(dfs)

    # keep the global result once (don't overwrite it when the refinement is re-run)
    src = os.path.join(path, f"Result_calibration{add_name}.json")
    dst = os.path.join(path, f"Result_calibration{add_name}_global.json")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    param_glob = np.array(fm.output.read_from_json(f"Result_calibration{add_name}_global.json", dir=path)["param_ode"])

    start = time.time()
    param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup,
                                                     maxiter=MAXITER, n_restarts=N_RESTARTS, jitter_frac=JITTER_FRAC, n_jobs=N_JOBS,
                                                     history_file=path + "optimization_history_local.csv")
    print((time.time() - start) / 60., "min")

    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    fm.output.json_dump({"param_ode": param_opt.astype(list), "s_x": s_x, "T_x": result["T_x"], "cost": cost_opt},
                        f"Result_calibration{add_name}.json", dir=path)
    print(f"Saved refined optimum to {src}")

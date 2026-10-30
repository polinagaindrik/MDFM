"""
Local refinement of existing calibration results
================================================

Runs the local step (fm.pest.local_optimization) on calibrations that were already
obtained with the global optimizer, without repeating the global search. Loops over
all runs of the studies in STUDIES (see paper_paths.py), e.g. first model_complexity,
then noise_vs_nspecies. The local setup (objective, noise model, penalty, method,
backend, restarts) is the same as in test_2sp.py.
FIX_PARAMS holds chosen parameters at fixed values (e.g. non-identifiable ones at 0)
and refines only the others.

Per run (<path> = .../calibration/):
Reads   <path>/Result_calibration<add_name>_global.json  (start; created from
        Result_calibration<add_name>.json if missing) + data frames
Writes  <path>/Result_calibration<add_name>.json         (refined optimum; an existing
        result with another objective is kept as Result_calibration<add_name>_<objective>.json)
        <path>/optimization_history_local.csv, optimization_history_local_start<r>.csv (written live; the
        files of an earlier refinement are moved to optimization_history_local_prev/)
        <path>/Result_calibration<add_name>_partial.json  (best finished start so far, while the run is going)

Run from the repository root:
    python model_paper/main_paper_local_refine.py
"""
import os
import re
import sys
import glob
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import (load_calibration, s_x_predefined_from_maldi, exp_temps_from_columns,
                                            make_param_names, resolve_fixed_values, print_parameter_table,
                                            copy_result_to_all_profiles)
from test_2sp import rescale_truth_max_s
import paper_paths as pp

import numpy as np
import time

# standard run folders of paper_paths.run_dir, e.g. '4_dim_2media_exp_10noise'
_STANDARD_RUN = re.compile(r'^\d+_dim_\d+media_exp_\d+noise$')
_RESULT_FILE = re.compile(r'^Result_calibration(_(\d+)dim_(\d+)media)(?:_global)?\.json$')


def find_runs(study, include_nonstandard=False, n_cl_list=None, noise_list=None):
    """
    Calibration folders of a study with a global result: list of dicts (path, add_name, n_cl, n_media,
    relnoise), sorted by noise level and number of species. include_nonstandard: also folders such as
    '10_dim_additional' or '8_dim_2media_exp_0noise_old'. n_cl_list / noise_list: only these values (None: all).
    """
    runs = {}
    for f in glob.glob(os.path.join(pp.study_dir(study), '**', 'calibration', 'Result_calibration*.json'),
                       recursive=True):
        m = _RESULT_FILE.match(os.path.basename(f))
        if m is None:
            continue
        path = os.path.dirname(f) + '/'
        run_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if not include_nonstandard and not _STANDARD_RUN.match(run_name):
            continue
        noise = re.search(r'(\d+)noise', os.path.relpath(path, pp.study_dir(study)))
        relnoise = int(noise.group(1)) / 100. if noise else 0.1  # model_complexity: 10 % noise
        n_cl = int(m.group(2))
        if (n_cl_list is not None and n_cl not in n_cl_list) or (noise_list is not None and relnoise not in noise_list):
            continue
        runs[path] = {'path': path, 'add_name': m.group(1), 'n_cl': n_cl, 'n_media': int(m.group(3)),
                      'relnoise': relnoise}
    return sorted(runs.values(), key=lambda r: (r['relnoise'], r['n_cl'], r['path']))


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    # Studies to refine, in this order (see paper_paths.py), e.g. ['model_complexity', 'noise_vs_nspecies']
    STUDIES = ['model_complexity']
    INCLUDE_NONSTANDARD = False  # True: also folders like '10_dim_additional', '..._old'
    N_CL_LIST = None             # e.g. [4, 8]: only these numbers of species (None: all)
    NOISE_LIST = None            # e.g. [0.0, 0.2]: only these noise levels (None: all)
    # True: skip runs whose Result_calibration*.json already has this objective / noise model / backend
    SKIP_DONE = True
    n_jobs = 22  # local multistart processes per run (at most N_RESTARTS are used)
    # runs refined at the same time; N_PARALLEL_RUNS * min(n_jobs, N_RESTARTS) a bit above the number of cores
    # keeps all cores busy while runs wait for their slowest starts (output of the runs is interleaved)
    N_PARALLEL_RUNS = 3
    # Every start writes its history live to optimization_history_local_start<r>.csv and the best finished start
    # is saved at once to Result_calibration<add_name>_partial.json (removed when the run is complete).
    # True: a run that was interrupted (its _partial.json with the same settings exists) restarts from the best
    # point of its saved histories instead of the global result
    RESUME_INTERRUPTED = True
    # Ending slow runs early (the best points so far are always kept and saved):
    # N_AGREE: stop the remaining starts of a run as soon as N_AGREE finished starts reach the best cost within
    # AGREE_RTOL (relative); None: wait for all starts
    N_AGREE = 3
    AGREE_RTOL = 1e-5
    MAX_MINUTES_PER_START = None  # time limit per start, e.g. 600; None: no limit
    # by hand while it runs: touch <path>/optimization_history_local.stop -> all starts of that run stop at their
    # next Jacobian and the run is saved as usual

    # local step as in test_2sp.py
    # 'L-BFGS-B' or 'least_squares' (trust-region least squares on the residuals, parameters scaled, much faster
    # convergence for ill-conditioned problems)
    LOCAL_METHOD = 'least_squares'
    MAXITER = 500 if LOCAL_METHOD == 'least_squares' else 1000
    N_RESTARTS = 10      # start 0 = global result, others = random perturbations of it
    JITTER_FRAC = 0.1   # size of the perturbations (fraction of the bound width)
    # derivatives of the fits: 'numpy' (finite differences) or 'jax' (exact derivatives by automatic
    # differentiation; needs: pip install jax diffrax). Don't mix backends within one profile run.
    BACKEND = 'jax'
    # Parameters held at a fixed value (same in every run), e.g. {'k_00,01': 0.}; stored as 'fixed_params'
    FIX_PARAMS = {}
    # Penalty so that the largest media coefficient S is 1 (fixes the x0/S scale trade-off): weight w, the cost
    # rises by the factor exp(w * (max S - 1)^2). None: off.
    S_MAX_PENALTY = 100
    # True (with S_MAX_PENALTY): rescale the global result to max(S) = 1 before the refinement (S / m, x0 and
    # N_1 + log10(m), k / m; same observables), so that the penalty does not start far from its minimum
    RESCALE_START_MAX_S = True
    # Noise model of the residuals: 'absolute' or 'relative' (multiplicative noise, log residuals)
    NOISE_MODEL = 'relative'
    # Objective: 'neg2logL', 'geomean', 'calibration' or 'known_noise' (see test_2sp.py)
    OBJECTIVE = 'neg2logL'
    # noise-free data (relnoise = 0): neg2logL estimates the noise from the residuals and diverges, so these
    # runs use -2 log L with the known noise level KNOWN_SIGMA instead
    OBJECTIVE_NOISE_FREE = 'known_noise'
    KNOWN_SIGMA = 0.1
    # ================================================================

    runs = [r for study in STUDIES for r in find_runs(study, INCLUDE_NONSTANDARD, N_CL_LIST, NOISE_LIST)]
    print(f'{len(runs)} runs to refine:')
    for r in runs:
        print(f"  {r['path']}  ({r['n_cl']} species, {r['n_media']} media, noise {r['relnoise']:g})")

    def refine(r):
        """Refine one run; returns its summary line (path, status, cost_start, cost_opt, minutes)."""
        path, add_name, n_cl, n_media = r['path'], r['add_name'], r['n_cl'], r['n_media']
        objective = OBJECTIVE_NOISE_FREE if r['relnoise'] == 0 else OBJECTIVE
        print(f"\n{'=' * 70}\n{path}  (objective {objective}, noise model {NOISE_MODEL})\n{'=' * 70}")
        src = os.path.join(path, f"Result_calibration{add_name}.json")
        dst = os.path.join(path, f"Result_calibration{add_name}_global.json")
        if SKIP_DONE and os.path.exists(src):
            old = fm.output.read_from_json(f"Result_calibration{add_name}.json", dir=path)
            if (old.get('objective') == objective and old.get('noise_model') == NOISE_MODEL
                    and old.get('backend') == BACKEND):
                print('Already refined with these settings, skipped (SKIP_DONE)')
                return path, 'skipped', None, None, None
        try:
            # start: Result_calibration<add_name>_global.json if present, otherwise the saved
            # Result_calibration<add_name>.json, kept as _global.json (it is overwritten by the refined optimum)
            if os.path.exists(dst):
                print(f'Start: {dst}')
            else:
                shutil.copy(src, dst)
                print(f'Start: {src} (no _global.json; copied to {dst})')
            dfs, result, _ = load_calibration(path, add_name)
            df_mibi, df_maldi, df_ngs = dfs
            exps = sorted(list(set([s.split("_")[0] for s in df_mibi.columns])))
            media = sorted(list(set([s.split("_")[-1].split("-")[0] for s in df_maldi.columns])))
            result_glob = fm.output.read_from_json(f"Result_calibration{add_name}_global.json", dir=path)
            T_x = result_glob.get("T_x", result.get("T_x"))

            calibr_presetup = {
                "model": fm.mdl.fusion_model2,
                "T_x": T_x,
                "output_path": path,
                "n_cl": n_cl,
                "n_media": n_media,
                "dfs": dfs,
                "exps": exps,
                "exp_temps": exp_temps_from_columns(df_mibi),
                "media": media,
            }
            calibr_setup = fm.pest.define_calibr_setup_insilico(
                calibr_presetup, inhib=True, s_x_predefined=s_x_predefined_from_maldi(df_maldi, media), s_x=None)
            calibr_setup["data_array"] = fm.dtf.extract_observables_from_df(dfs)
            calibr_setup["s_max_penalty"] = S_MAX_PENALTY
            calibr_setup["noise_model"] = NOISE_MODEL
            calibr_setup["aggregation_func"] = {
                'neg2logL': lambda: pl.make_gaussian_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'geomean': lambda: pl.make_geomean_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'calibration': lambda: fm.pest.cost_sum_and_geometric_mean,
                'known_noise': lambda: pl.make_known_noise_neg2loglik(calibr_setup['data_array'], KNOWN_SIGMA,
                                                                      NOISE_MODEL)}[objective]()

            param_glob = np.array(result_glob["param_ode"], dtype=float)
            if S_MAX_PENALTY and RESCALE_START_MAX_S:
                m_s = float(np.max(param_glob[-n_cl * n_media:]))
                param_glob = rescale_truth_max_s(param_glob, n_cl, len(exps), n_media)
                bnds = np.array(calibr_setup["param_bnds"], dtype=float)
                param_glob = np.clip(param_glob, bnds[:, 0], bnds[:, 1])
                print(f'Start rescaled to max(S) = 1 (factor {m_s:g})')
            names = make_param_names(exps, list(df_maldi.index), media)
            fixed = resolve_fixed_values(FIX_PARAMS, names)
            if fixed:
                print_parameter_table(names, param_glob, calibr_setup["param_bnds"], fixed)
            hist = path + "optimization_history_local.csv"
            partial = f"Result_calibration{add_name}_partial.json"
            settings = {"objective": objective, "noise_model": NOISE_MODEL, "backend": BACKEND,
                        "s_max_penalty": S_MAX_PENALTY, "fixed_params": fm.pest.fixed_values_record(fixed, names)}
            if RESUME_INTERRUPTED and os.path.exists(path + partial):
                old = fm.output.read_from_json(partial, dir=path)
                rec = fm.pest.best_from_history(hist)
                if all(old.get(k) == v for k, v in settings.items()) and rec is not None:
                    print(f'Interrupted run found: resuming from start {rec[2]} of its history (cost {rec[1]:.6g})')
                    param_glob = np.array(rec[0], dtype=float)
            # keep the histories of an earlier refinement; the new starts overwrite these files live
            old_hist = glob.glob(path + "optimization_history_local*.csv")
            if old_hist:
                prev = path + "optimization_history_local_prev/"
                os.makedirs(prev, exist_ok=True)
                for f in old_hist:
                    shutil.move(f, prev + os.path.basename(f))
                print(f'Previous histories moved to {prev}')
            cost_start = fm.pest.cost_withS(param_glob, calibr_setup, None)
            print(f'Cost ({objective}) at the start: {cost_start:.6g}')

            def checkpoint(param, cost, r, n_done, n_total):
                """Save the best finished start so far, so it is kept if the run is interrupted."""
                fm.output.json_dump({"param_ode": np.asarray(param).astype(list), "T_x": T_x, "cost": cost,
                                     "start": r, "starts_done": f"{n_done}/{n_total}", **settings},
                                    partial, dir=path)
                print(f'[{os.path.basename(os.path.dirname(os.path.dirname(path)))}] start {r} done '
                      f'({n_done}/{n_total}), new best cost {cost:.6g} saved to {path + partial}', flush=True)
            fm.output.json_dump({"param_ode": param_glob.astype(list), "T_x": T_x, "cost": cost_start,
                                 "start": None, "starts_done": "0", **settings}, partial, dir=path)

            start = time.time()
            param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup,
                                                             fixed_values=fixed, maxiter=MAXITER,
                                                             n_restarts=N_RESTARTS, jitter_frac=JITTER_FRAC,
                                                             n_jobs=n_jobs, method=LOCAL_METHOD, backend=BACKEND,
                                                             history_file=hist, checkpoint_func=checkpoint,
                                                             n_agree=N_AGREE, agree_rtol=AGREE_RTOL,
                                                             max_minutes=MAX_MINUTES_PER_START)
            minutes = (time.time() - start) / 60.
            print(f'{minutes:.1f} min, cost = {cost_opt:.6g}')

            # keep an existing refined result with another objective / noise model (as in test_2sp.py)
            if os.path.exists(src):
                old = fm.output.read_from_json(f"Result_calibration{add_name}.json", dir=path)
                old_nm = old.get('noise_model', 'absolute')
                old_obj = old.get('objective', 'calibration') + ('' if old_nm == 'absolute' else f'_{old_nm}')
                if old_obj != objective + ('' if NOISE_MODEL == 'absolute' else f'_{NOISE_MODEL}'):
                    bak = src.replace('.json', f'_{old_obj}.json')
                    if not os.path.exists(bak):
                        shutil.copy(src, bak)
                        print(f'Kept the previous result (objective {old_obj}) as {bak}')
            s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
            fm.output.json_dump({"param_ode": param_opt.astype(list), "s_x": s_x, "T_x": T_x, "cost": cost_opt,
                                 "fixed_params": fm.pest.fixed_values_record(fixed, names),
                                 "s_max_penalty": S_MAX_PENALTY, "objective": objective,
                                 "noise_model": NOISE_MODEL, "backend": BACKEND},
                                f"Result_calibration{add_name}.json", dir=path)
            print(f"Saved refined optimum to {src}")
            copy_result_to_all_profiles(path, add_name)  # new optimum into existing profile JSON files
            os.remove(path + partial)  # run complete
            return path, 'refined', cost_start, cost_opt, minutes
        except Exception:
            traceback.print_exc()
            print(f'Refinement of {path} failed, continuing with the next run')
            return path, 'failed', None, None, None

    # Several runs at once (threads; each run starts its own pool of min(n_jobs, N_RESTARTS) processes), so
    # cores left idle while one run waits for its slowest starts are used by the starts of other runs
    print(f'{N_PARALLEL_RUNS} runs in parallel, up to {N_PARALLEL_RUNS * min(n_jobs, N_RESTARTS)} processes')
    with ThreadPoolExecutor(max_workers=N_PARALLEL_RUNS) as ex:
        summary = list(ex.map(refine, runs))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for path, status, c0, c1, minutes in summary:
        costs = f'cost {c0:.6g} -> {c1:.6g}, {minutes:.1f} min' if status == 'refined' else ''
        print(f'  {status:8s} {path}  {costs}')

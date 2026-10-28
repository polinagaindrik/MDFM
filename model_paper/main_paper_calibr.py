"""
Calibration of the model paper runs (in-silico data)
====================================================

Same stages and settings as test_2sp.py, for the paper studies (paper_paths.py):

  1) in-silico data: fusion_model2 with the true parameters of n_cl species, experimental design DESIGN
  2) global optimization (GLOBAL_METHOD)               -> Result_calibration<add_name>_global.json
  3) local refinement (LOCAL_METHOD multistart)        -> Result_calibration<add_name>.json

Each stage can be switched off (RUN_*) and is then read from its saved result. The profile likelihood
is computed by profile_likelihood_model_paper.py. Run from the repository root:
    python model_paper/main_paper_calibr.py
"""
import os
import sys
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
import paper_paths as pp
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import (make_param_names, exp_temps_from_columns, resolve_fixed_values,
                                            print_parameter_table, copy_result_to_all_profiles)
from test_2sp import DESIGNS

import numpy as np


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    # Output folder (see paper_paths.py): 'model_complexity', 'noise_vs_nspecies' or 'media_influence'
    STUDY = 'media_influence'
    # Experimental design of the in-silico data (test_2sp.DESIGNS). The paper runs use 'rep3' (pp.PAPER_DESIGN,
    # usual folders); other designs get their own run folder <run>_<design>/. After switching: RUN_DATA = True.
    DESIGN = 'rep3'
    temps = DESIGNS[DESIGN]['temps']
    days_meas = DESIGNS[DESIGN].get('days')  # None: default 0..17
    n_cl = 6
    n_media = 2
    relnoise = 0.1
    ntr = 1                        # keep 1: replicates via repeated temperatures (see DESIGNS)
    MEDIA = 'sel1'  # media_influence only: 'gen1', ..., 'sel3', 'gen1+sel1', 'gen2+sel2', ... (sets n_media and GEN_MODEL)
    workers = -1                   # differential evolution: -1 = all cores
    n_jobs = 22                    # local multistart processes

    RUN_DATA = True                # 1) generate the in-silico data
    RUN_GLOBAL = True              # 2) global optimization
    RUN_LOCAL = True               # 3) local refinement

    # optimizer settings
    # global step: 'de'            differential evolution (as before)
    #              'multistart'    N_STARTS local fits from Latin hypercube points (recommended for many species)
    #              'de+multistart' differential evolution, then the multistart incl. the DE result
    GLOBAL_METHOD = 'de'
    N_STARTS = 1
    # local step: 'L-BFGS-B' (as before) or 'least_squares' (trust-region least squares on the residuals,
    # parameters scaled, much faster convergence for ill-conditioned problems)
    LOCAL_METHOD = 'least_squares'
    # derivatives of the fits: 'numpy' (finite differences, as before) or 'jax' (exact derivatives by automatic
    # differentiation, much faster and a more accurate ODE solve; needs: pip install jax diffrax). Used for the
    # local step and multistart (not for differential evolution). Don't mix backends within one
    # profile run: the profile tools check the cost at the stored optimum and stop if it differs.
    BACKEND = 'jax'
    # local step: parameters held at a fixed value, only the others are refined (e.g. non-identifiable ones
    # set to 0). Keys: index or name (see the printed parameter table), e.g. {17: 0., 18: 0.} or
    # {'k_00,01': 0.}; the value may lie outside the bounds. Stored as 'fixed_params' in
    # Result_calibration*.json and kept fixed in the profile likelihood (not profiled, not re-fitted).
    FIX_PARAMS = {}
    # True: FIX_PARAMS also in the global step (search with these parameters fixed)
    FIX_IN_GLOBAL = False
    # Penalty so that the largest media coefficient S is 1 (fixes the x0/S scale trade-off): weight w, the cost
    # rises by the factor exp(w * (max S - 1)^2), e.g. w = 100: max S = 0.9 -> +172 %, 0.97 -> +9 %. None: off.
    # Stored in the result JSON; the profile likelihood uses the same penalty (needs new profiles).
    S_MAX_PENALTY = 100
    # Noise model of the residuals (all optimizations and the profiles):
    #   'absolute': measured - model (MiBi divided by its maximum), as before
    #   'relative': multiplicative noise (measured = model * (1 + eps)): log residuals for MiBi, centred
    #               log-ratios within each composition for MALDI / NGS -> constant variance, valid chi2 threshold
    NOISE_MODEL = 'relative'
    # Objective of all optimizations (stored as 'objective' in the result JSON files; an existing result with a
    # different objective is kept as Result_calibration<...>_<objective>.json):
    # 'neg2logL'   : -2 log L, data types weighted by their number of points
    # 'geomean'    : -2 log L form of cost_sum_and_geometric_mean (same optimum, equal weight per data type)
    # 'calibration': cost_sum_and_geometric_mean itself (as before)
    # 'known_noise': -2 log L with a KNOWN noise level KNOWN_SIGMA, e.g. for noise-free data (relnoise = 0);
    #                the objectives above estimate the noise from the residuals and diverge for noise-free data.
    CALIBRATION_OBJECTIVE = 'neg2logL' if relnoise > 0 else 'known_noise'
    # noise level for 'known_noise': one value or [NGS, MALDI, MiBi], in the units of the residuals
    # (NOISE_MODEL = 'relative': standard deviation of the log residuals, 0.1 ~ 10 % relative noise)
    KNOWN_SIGMA = 0.1
    # ================================================================

    media_run = MEDIA if STUDY == 'media_influence' else None
    if media_run is not None:
        n_media = pp.media_n(media_run)
    add_name = pp.add_name(n_cl, n_media, media=media_run)
    path = pp.ensure_dir(pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=media_run,
                                       design=DESIGN))
    print(f'Design {DESIGN!r}: temperatures {temps}, days {days_meas if days_meas is not None else "0..17"}, '
          f'output {path} (files *{add_name}*)')
    start = time.time()

    # 1) in-silico data -------------------------------------------------
    # data generator = media of the in-silico data, must match n_media (functions in fusion_model/data/insilico_models.py):
    #   1 medium : fm.data.model_1media_exp_sel1 / _sel2 / _sel3 / _gen1 / _gen2 / _gen3
    #   2 media  : fm.data.model_2media_expgensel (sel1 + gen1), _expgen2sel2, _expgen3sel3, _expsel1sel2, ...
    #   media_influence: chosen from MEDIA automatically
    GEN_MODEL = fm.data.model_2media_expgensel
    if media_run is not None:
        GEN_MODEL = getattr(fm.data, pp.generator_name(media_run))
    # setup files in model_paper/setup/: initial values, media matrix and true parameters
    # (True_parameters_paper.json, created from the ZL2030 estimates by main_true_parameters.py)
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=fm.data.SETUP_DIR)
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=fm.data.SETUP_DIR)['x0'])[:len(temps), :n_cl]
    if len(x10) < len(temps):
        raise ValueError(f'design {DESIGN!r} has {len(temps)} experiments, the setup has initial values for {len(x10)}')
    param_ode_true = fm.data.read_true_parameters(n_cl)['param_ode']
    print(f'True ODE parameters ({n_cl} species) from {fm.data.SETUP_DIR}{fm.data.TRUE_PARAMETERS_FILE}:', param_ode_true)
    if RUN_DATA or not os.path.exists(path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            GEN_MODEL, n_cl, temps, ntr, S_matrix_setup, param_ode=param_ode_true.copy(),
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=path, add_name=add_name,
            t_meas=days_meas)
    else:
        import pandas as pd
        dfs = [pd.read_pickle(path + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
        _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
            dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    df_mibi, df_maldi, df_ngs = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    if len(media) != n_media:
        raise ValueError(f'GEN_MODEL {GEN_MODEL.__name__} produces {len(media)} media {media}, but n_media = {n_media}')
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    bact = list(df_maldi.index)
    print(f'Data: {len(exps)} experiments {exps}, media {media}, species {bact}')

    # calibration setup ---------------------------------------------------
    calibr_presetup = {
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers,
        'output_path': path,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(df_mibi),
        'media': media,
    }
    print('Temperatures:', calibr_presetup['exp_temps'])
    fm.data.save_exp_temps(calibr_presetup['exp_temps'], path)  # next to the results (output directory)
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    calibr_setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    calibr_setup['s_max_penalty'] = S_MAX_PENALTY
    calibr_setup['noise_model'] = NOISE_MODEL

    def make_objective(name):
        """Aggregation function of an objective name (see CALIBRATION_OBJECTIVE)."""
        return {'neg2logL': pl.make_gaussian_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'geomean': pl.make_geomean_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'calibration': fm.pest.cost_sum_and_geometric_mean,
                'known_noise': pl.make_known_noise_neg2loglik(calibr_setup['data_array'], KNOWN_SIGMA, NOISE_MODEL)}[name]

    def save_result(fname, data):
        """Write a result JSON with its objective and noise model; keep an existing file that has a different
        objective or noise model (as <file>_<objective>[_relative].json)."""
        f = os.path.join(path, fname)
        if os.path.exists(f):
            old = fm.output.read_from_json(fname, dir=path)
            old_nm = old.get('noise_model', 'absolute')
            old_obj = old.get('objective', 'calibration') + ('' if old_nm == 'absolute' else f'_{old_nm}')
            if old_obj != CALIBRATION_OBJECTIVE + ('' if NOISE_MODEL == 'absolute' else f'_{NOISE_MODEL}'):
                bak = f.replace('.json', f'_{old_obj}.json')
                if not os.path.exists(bak):
                    import shutil
                    shutil.copy(f, bak)
                    print(f'Kept the previous result (objective {old_obj}) as {bak}')
        fm.output.json_dump(dict(data, objective=CALIBRATION_OBJECTIVE, noise_model=NOISE_MODEL, backend=BACKEND),
                            fname, dir=path)

    calibr_setup['aggregation_func'] = make_objective(CALIBRATION_OBJECTIVE)
    print(f'Objective of the optimizations: {CALIBRATION_OBJECTIVE}, noise model: {NOISE_MODEL}')
    names = make_param_names(exps, bact, media)

    # 2) global optimization ---------------------------------------------
    fm.pest.optimization.output_file = path + 'optimization_history1.csv'  # DE history into the run folder
    fixed_glob = resolve_fixed_values(FIX_PARAMS, names) if FIX_IN_GLOBAL else {}
    if RUN_GLOBAL:
        glob_setup = calibr_setup
        if fixed_glob:
            print_parameter_table(names, [b[0] for b in calibr_setup['param_bnds']], calibr_setup['param_bnds'], fixed_glob)
            glob_setup = fm.pest.apply_fixed_values(calibr_setup, fixed_glob, names=names)
        de_starts = []
        if GLOBAL_METHOD in ('de', 'de+multistart'):
            param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_withS, glob_setup)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            de_starts = [param_glob]
        if GLOBAL_METHOD in ('multistart', 'de+multistart'):
            param_glob, cost_glob = fm.pest.multistart_optimization(
                fm.pest.cost_withS, glob_setup, n_starts=N_STARTS, extra_starts=de_starts, method=LOCAL_METHOD,
                backend=BACKEND,
                maxiter=200 if LOCAL_METHOD == 'least_squares' else 500, n_jobs=n_jobs,
                history_file=path + 'optimization_history_multistart.csv')
        elif GLOBAL_METHOD != 'de':
            raise ValueError("GLOBAL_METHOD must be 'de', 'multistart' or 'de+multistart'")
        print(f'{(time.time() - start) / 60.:.1f} min (global), cost = {cost_glob:.6g}')
        s_x_glob = np.array(param_glob)[-n_cl*n_media:].reshape((n_media, n_cl))
        save_result(f'Result_calibration{add_name}_global.json',
                    {'param_ode': param_glob.astype(list), 's_x': s_x_glob, 'T_x': T_x, 'cost': cost_glob,
                     'fixed_params': fm.pest.fixed_values_record(fixed_glob, names), 's_max_penalty': S_MAX_PENALTY})
    else:
        param_glob = np.array(fm.output.read_from_json(f'Result_calibration{add_name}_global.json', dir=path)['param_ode'])

    # 3) local refinement ------------------------------------------------
    if RUN_LOCAL:
        fixed = resolve_fixed_values(FIX_PARAMS, names)
        if fixed:
            print_parameter_table(names, param_glob, calibr_setup['param_bnds'], fixed)
        param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup, fixed_values=fixed,
                                                         maxiter=400 if LOCAL_METHOD == 'least_squares' else 1000,
                                                         n_restarts=5, jitter_frac=0.1, n_jobs=n_jobs, method=LOCAL_METHOD,
                                                         backend=BACKEND, history_file=path + 'optimization_history_local.csv')
        print(f'{(time.time() - start) / 60.:.1f} min (global + local), cost = {cost_opt:.6g}')
        s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
        # final (refined) result keeps the usual file name, so all downstream scripts use it
        save_result(f'Result_calibration{add_name}.json',
                    {'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                     'fixed_params': fm.pest.fixed_values_record(fixed, names), 's_max_penalty': S_MAX_PENALTY})
        copy_result_to_all_profiles(path, add_name)  # new optimum into existing profile JSON files
    print(f'{(time.time() - start) / 60.:.1f} min in total')

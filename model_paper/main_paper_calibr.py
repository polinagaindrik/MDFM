import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from profile_likelihood_model_paper import (make_param_names, resolve_fixed_values, print_parameter_table,
                                            copy_result_to_all_profiles)
import paper_paths as pp

import numpy as np
import time

if __name__ == "__main__":
    # Output folder (see paper_paths.py): 'model_complexity', 'noise_vs_nspecies' or 'media_influence'
    STUDY = 'media_influence'
    workers = -1
    n_cl = 6
    n_media = 2
    relnoise = 0.1
    MEDIA = 'sel1'  # media_influence only: 'gen1', ..., 'sel3', 'gen1+sel1', 'gen2+sel2', ... (sets n_media and GEN_MODEL)
    # Parameters held at a fixed value, e.g. non-identifiable ones at 0 (keys: index or name, value may lie
    # outside the bounds), e.g. {'k_00,01': 0.}. Applied in the local refinement; with FIX_IN_GLOBAL = True
    # also in the global search. Stored as 'fixed_params' in the result JSON (profiles keep them fixed).
    FIX_PARAMS = {}
    FIX_IN_GLOBAL = False
    # Penalty so that the largest media coefficient S is 1 (fixes the x0/S scale trade-off): weight w, the cost
    # rises by the factor exp(w * (max S - 1)^2), e.g. w = 100: max S = 0.9 -> +172 %, 0.97 -> +9 %. None: off.
    # Stored in the result JSON; the profile likelihood uses the same penalty (needs new profiles).
    S_MAX_PENALTY = None
    # Noise model of the residuals (all optimizations and the profiles):
    #   'absolute': measured - model (MiBi divided by its maximum), as before
    #   'relative': multiplicative noise (measured = model * (1 + eps)): log residuals for MiBi, centred
    #               log-ratios within each composition for MALDI / NGS -> constant variance, valid chi2 threshold
    NOISE_MODEL = 'absolute'

    media_run = MEDIA if STUDY == 'media_influence' else None
    if media_run is not None:
        n_media = pp.media_n(media_run)
    add_name = pp.add_name(n_cl, n_media, media=media_run)
    path_new = pp.ensure_dir(pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=media_run))
    print(f'Output: {path_new} (files *{add_name}*)')

    # 'Real' model:
    # data generator = media of the in-silico data, must match n_media (functions in fusion_model/data/insilico_models.py):
    #   1 medium : fm.data.model_1media_exp_sel1 / _sel2 / _sel3 / _gen1 / _gen2 / _gen3
    #   2 media  : fm.data.model_2media_expgensel (sel1 + gen1), _expgen2sel2, _expgen3sel3, _expsel1sel2, ...
    #   media_influence: chosen from MEDIA automatically
    GEN_MODEL = fm.data.model_2media_expgensel
    if media_run is not None:
        GEN_MODEL = getattr(fm.data, pp.generator_name(media_run))
    temps = [2., 2., 2., 6., 6., 6., 10., 10., 10.]
    ntr = 1
    # setup files in model_paper/setup/: initial values, media matrix and true parameters
    # (True_parameters_paper.json, created from the ZL2030 estimates by main_true_parameters.py)
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=fm.data.SETUP_DIR)['x0'])[:len(temps), :n_cl]
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=fm.data.SETUP_DIR)
    param_ode_true = fm.data.read_true_parameters(n_cl)['param_ode']
    print(f'True ODE parameters ({n_cl} species) from {fm.data.SETUP_DIR}{fm.data.TRUE_PARAMETERS_FILE}:', param_ode_true)
    dfs_calibr, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(GEN_MODEL, n_cl, temps, ntr, S_matrix_setup, x10=x10,
                                                                              param_ode=param_ode_true.copy(),
                                                                              inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0.,
                                                                              path=path_new, add_name=add_name)
    (df_mibi, df_maldi, df_ngs) = dfs_calibr
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    if len(media) != n_media:
        raise ValueError(f'GEN_MODEL {GEN_MODEL.__name__} produces {len(media)} media {media}, but n_media = {n_media}')
    #n_cl = np.shape(dfs_calibr[1])[0] 
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))

    start = time.time()
    calibr_presetup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_new,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps_calibr,
        'exp_temps': {exp: temp for exp, temp in zip(exps_calibr, temps)},
        'media': media, 
        }
    fm.data.save_exp_temps(calibr_presetup['exp_temps'], path_new)  # next to the results (output directory)
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    calibr_setup['s_max_penalty'] = S_MAX_PENALTY
    calibr_setup['noise_model'] = NOISE_MODEL
    names = make_param_names(exps_calibr, list(df_maldi.index), media)
    fixed = resolve_fixed_values(FIX_PARAMS, names)
    fixed_glob = fixed if FIX_IN_GLOBAL else {}
    if fixed:
        print_parameter_table(names, [b[0] for b in calibr_setup['param_bnds']], calibr_setup['param_bnds'], fixed)

    # 1) Global optimization (differential evolution)
    glob_setup = fm.pest.apply_fixed_values(calibr_setup, fixed_glob, names=names) if fixed_glob else calibr_setup
    param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_withS, glob_setup)
    calibr_setup['data_array'] = glob_setup['data_array']
    print((time.time()-start)/60., 'min (global)')
    s_x_glob = np.array(param_glob)[-n_cl*n_media:].reshape((n_media, n_cl))
    fm.output.json_dump({'param_ode': param_glob.astype(list), 's_x': s_x_glob, 'T_x': T_x, 'cost': cost_glob,
                         'fixed_params': fm.pest.fixed_values_record(fixed_glob, names),
                         's_max_penalty': S_MAX_PENALTY, 'noise_model': NOISE_MODEL},
                        f'Result_calibration{add_name}_global.json', dir=path_new)

    # 2) Local refinement from the global result -> exact local optimum
    #    LOCAL_METHOD: 'L-BFGS-B' (as before) or 'least_squares' (trust-region least squares on the residuals,
    #    parameters scaled; much faster convergence for ill-conditioned problems, maxiter = evaluations per solve)
    #    n_restarts: start 0 = global result, the others = random perturbations of it (jitter_frac*bound width)
    #    n_jobs: starts run in parallel processes (-1 = all cores)
    LOCAL_METHOD = 'L-BFGS-B'
    # derivatives of the fits: 'numpy' (finite differences, as before) or 'jax' (exact derivatives by automatic
    # differentiation, much faster and a more accurate ODE solve; needs: pip install jax diffrax). Used for the
    # local step, multistart and the profiles (not for differential evolution). Don't mix backends within one
    # profile run: the profile tools check the cost at the stored optimum and stop if it differs.
    BACKEND = 'numpy'
    param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup,
                                                     maxiter=200 if LOCAL_METHOD == 'least_squares' else 1000,
                                                     n_restarts=4, jitter_frac=0.02, n_jobs=4, method=LOCAL_METHOD, backend=BACKEND,
                                                     fixed_values=fixed,
                                                     history_file=path_new+'optimization_history_local.csv')
    print((time.time()-start)/60., 'min (global + local)')

    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    calibr_setup['s_x'] = s_x

    # Final (refined) result keeps the usual file name, so all downstream scripts use it
    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                         'fixed_params': fm.pest.fixed_values_record(fixed, names),
                         's_max_penalty': S_MAX_PENALTY, 'noise_model': NOISE_MODEL},
                        f'Result_calibration{add_name}.json', dir=path_new)
    copy_result_to_all_profiles(path_new, add_name)  # new optimum into existing profile JSON files
    param_opt = fm.output.read_from_json(''+f'Result_calibration{add_name}.json', dir=path_new)['param_ode']
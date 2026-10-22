import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from profile_likelihood_model_paper import make_param_names, resolve_fixed_values, print_parameter_table

import numpy as np
import time

if __name__ == "__main__":
    path = 'model_paper/out/'
    workers = -1
    n_cl = 6
    n_media = 1
    relnoise = 0.1
    # Parameters held at a fixed value, e.g. non-identifiable ones at 0 (keys: index or name, value may lie
    # outside the bounds), e.g. {'k_00,01': 0.}. Applied in the local refinement; with FIX_IN_GLOBAL = True
    # also in the global search. Stored as 'fixed_params' in the result JSON (profiles keep them fixed).
    FIX_PARAMS = {}
    FIX_IN_GLOBAL = False

    add_name = f'_{int(n_cl)}dim_{int(n_media)}media'
    path_new = path#path+f'noise_vs_nspecies/{int(relnoise*100)}noise/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/'
    #add_name = f'_{int(n_cl)}dim_{int(n_media)}media_gen'
    #path_new = path+f'{int (n_cl)}_dim_{n_media}media_gen/calibration/'

    # 'Real' model:
    temps = [2., 2., 2., 6., 6., 6., 10., 10., 10.]
    ntr = 1
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=path)['x0'])[:len(temps), :n_cl]
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=path)
    dfs_calibr, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(fm.data.model_2media_exp_sel, n_cl, temps, ntr, S_matrix_setup, x10=x10,
                                                                              inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0.,
                                                                              path=path_new, add_name=add_name)
    (df_mibi, df_maldi, df_ngs) = dfs_calibr
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
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
    fm.output.json_dump(calibr_presetup['exp_temps'], 'exp_temps_model_paper.json', dir='model_paper/')
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
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
                         'fixed_params': fm.pest.fixed_values_record(fixed_glob, names)},
                        f'Result_calibration{add_name}_global.json', dir=path_new)

    # 2) Local refinement from the global result -> exact local optimum
    #    LOCAL_METHOD: 'L-BFGS-B' (as before) or 'least_squares' (trust-region least squares on the residuals,
    #    parameters scaled; much faster convergence for ill-conditioned problems, maxiter = evaluations per solve)
    #    n_restarts: start 0 = global result, the others = random perturbations of it (jitter_frac*bound width)
    #    n_jobs: starts run in parallel processes (-1 = all cores)
    LOCAL_METHOD = 'L-BFGS-B'
    param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup,
                                                     maxiter=200 if LOCAL_METHOD == 'least_squares' else 1000,
                                                     n_restarts=4, jitter_frac=0.02, n_jobs=4, method=LOCAL_METHOD,
                                                     fixed_values=fixed,
                                                     history_file=path_new+'optimization_history_local.csv')
    print((time.time()-start)/60., 'min (global + local)')

    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    calibr_setup['s_x'] = s_x

    # Final (refined) result keeps the usual file name, so all downstream scripts use it
    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                         'fixed_params': fm.pest.fixed_values_record(fixed, names)},
                        f'Result_calibration{add_name}.json', dir=path_new)
    param_opt = fm.output.read_from_json(''+f'Result_calibration{add_name}.json', dir=path_new)['param_ode']
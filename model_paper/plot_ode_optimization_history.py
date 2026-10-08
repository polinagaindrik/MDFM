import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd


if __name__ == "__main__":
    n_cl = 4
    n_media = 2
    relnoise = 0.1

    #path = 'model_paper/out/'#f'model_paper/out/{int(n_cl)}_dim/calibration/'#
    #path2 = path+f'noise_vs_nspecies/{int(relnoise*100)}noise/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/'
    path = f'model_paper/out/model_complexity/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/'
    path2 = path
    add_name = f'_{int(n_cl)}dim_{int(n_media)}media'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    # 'json': final result Result_calibration<add_name>.json (after local refinement)
    # 'history': last step of the global optimization history (optimization_history1.csv)
    RESULT_SOURCE = 'json'
    os.makedirs(path2+'optimization/', exist_ok=True)

    # Cost function along the optimization (global and, if present, local refinement)
    optim_file2 = "optimization_history1.csv"
    if os.path.exists(path+optim_file2):
        df_optim2 = pd.read_csv(path+optim_file2)
        fm.plotting.plot_cost_function(df_optim2, path=path2+'optimization/')
    if os.path.exists(path+'optimization_history_local.csv'):
        df_optim_local = pd.read_csv(path+'optimization_history_local.csv')
        fm.plotting.plot_cost_function(df_optim_local, path=path2+'optimization/', add_name='_local')

    T_x = [1. for _ in range (n_cl)]
    if RESULT_SOURCE == 'json':
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'], dtype=float)
    else:
        # Take optimal parameter values on last optimization step
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    
    # Plot resulting model
    calibr_setup={
        'model': fm.mdl.fusion_model2,
    #    's_x': s_x, #param_real['s_x'],#read_from_json(path2+'S_matrix_true.json'),
        'T_x': T_x, #param_real['T_x'],#[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        'output_path': path2,
        'dfs': data,
        # temperature of each experiment from the column names, e.g. 'V01_M1_02C_00_MRS-mibi' -> 2.0
        'exp_temps': {c.split('_')[0]: float(c.split('_')[2].rstrip('C')) for c in data[0].columns},
        'media': sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns]))),
    }
    calibr_setup['s_x'] = s_x
    res_real = fm.output.read_from_json('Result_temp_together_real.json', dir=path2)
    param_ode_real = np.array(res_real['param_ode'])[n_cl*len(exps):]
    
    fm.plotting.plot_parameters(param_ode, bact_all, exps, clrs1, param_real=param_ode_real, path=path2+'optimization/')
    calibr_setup['dfs'] = data+[df_x]
    fm.plotting.plot_optimization_result(np.array(param_ode), calibr_setup, np.linspace(0, 17, 100),
                                         path=path2, clrs=clrs1, add_name=add_name+'_calibration')
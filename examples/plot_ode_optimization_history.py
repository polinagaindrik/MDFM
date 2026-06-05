import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd


if __name__ == "__main__":
    n_cl = 2
    n_media = 2
    relnoise = 0.1
    n_exps = 3

    path = f'out/main_param_distrib2/'
    path2 = path
    add_name = '_0'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
    data.append(df_x)

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    # ODE estimation:
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2+'optimization/')
    
    T_x = [1. for _ in range (n_cl)]
    # Take optimal parameter values on last optimization step
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    x0_vals = param_opt[:n_cl*len(exps)]
    lambd_opt = param_opt[n_cl*len(exps):n_cl*len(exps)+n_cl]
    alph_opt = param_opt[n_cl + n_cl*len(exps):n_cl + 2*n_cl*len(exps)]
    rest_ode_param = param_opt[n_cl + 2*n_cl*len(exps):]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    
    # Plot resulting model
    calibr_setup={
        'model': fm.mdl.fusion_model_distr,
    #    's_x': s_x, #param_real['s_x'],#read_from_json(path2+'S_matrix_true.json'),
        'T_x': T_x, #param_real['T_x'],#[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        'output_path': path2,
        'dfs': data,
        'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='out/main_param_distrib/'),
        'media': sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns]))),
    }
    calibr_setup['s_x'] = s_x
    #res_real = fm.output.read_from_json('Result_real_0.json', dir=path2)
    #param_ode_real = np.array(res_real['param_ode'])[n_cl*len(exps):]
    
    #fm.plotting.plot_parameters(param_ode, bact_all, exps, clrs1, param_real=param_ode_real, path=path2+'optimization/')
    #calibr_setup['dfs'] = data+[df_x]
    '''
    for i in range(n_exps):
        calibr_setup['dfs'] = fm.dtf.filter_dataframe(f'V{i:02d}', data)
        param_ode = np.concatenate([x0_vals, lambd_opt, alph_opt[n_cl*i:n_cl*(i+1)], rest_ode_param])
        fm.plotting.plot_optimization_result(np.array(param_ode), calibr_setup, np.linspace(0, 17, 100),
                                             path=path2, clrs=clrs1, add_name=add_name+'_calibration'+f'_{i}')
    '''
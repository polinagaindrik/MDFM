import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
import paper_paths as pp

import numpy as np
import pandas as pd


if __name__ == "__main__":
    # 'paper'   : calibration of one study (STUDY and n_cl etc. set below, folders see paper_paths.py)
    # 'test_2sp': 2-species test of model_paper/test_2sp.py
    # 'test_2sp_sepS': test_2sp_sepS.py (S estimated separately, then x0 + ODE with S fixed)
    # 'test_2sp_withS': test_2sp_withS.py (S estimated together with x0 + ODE, S-step cost terms added)
    CASE = 'test_2sp_withS'

    DESIGN = 'rep3'  # test_2sp* only: design as in test_2sp.py ('base', 'rep3', 'temps6', ...)
    NORMALIZE_TRUE_S = False  # test_2sp / test_2sp_withS only: data from the truth rescaled to max(S) = 1

    if CASE.startswith('test_2sp'):
        n_cl = 2
        n_media = 2
        relnoise = 0.1
        data_path = pp.test_2sp_path(DESIGN, relnoise, NORMALIZE_TRUE_S and CASE != 'test_2sp_sepS')
        # results of test_2sp_sepS / _withS in <data folder>_sepS/ resp. _withS/, data and truth in the data folder
        path = data_path if CASE == 'test_2sp' else data_path.rstrip('/') + CASE[len('test_2sp'):] + '/'
        add_name = f'_{int(n_cl)}dim_{int(n_media)}media_test'
        real_file = f'Result_real{add_name}.json'
    else:
        STUDY = 'model_complexity'  # 'model_complexity', 'noise_vs_nspecies' or 'media_influence'
        n_cl = 4
        n_media = 2
        relnoise = 0.1
        MEDIA = None  # media_influence only, e.g. 'gen2+sel2' (sets n_media)
        if MEDIA is not None:
            n_media = pp.media_n(MEDIA)
        path = pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=MEDIA)
        add_name = pp.add_name(n_cl, n_media, media=MEDIA)
        real_file = 'Result_temp_together_real.json'
        data_path = path
    path2 = path
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(data_path+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    df_x = pd.read_pickle(data_path+f'dataframe_x{add_name}.pkl')

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    # 'json': final result Result_calibration<add_name>.json (after local refinement)
    # 'history': last step of the global optimization history (optimization_history1.csv)
    RESULT_SOURCE = 'json'
    print(f'Plotting {CASE}: {path}, result from ' +
          (f'Result_calibration{add_name}.json' if RESULT_SOURCE == 'json' else 'optimization_history1.csv'))
    os.makedirs(path2+'optimization/', exist_ok=True)

    # Cost function along the optimization (global and, if present, local refinement)
    optim_file2 = "optimization_history1.csv"
    if os.path.exists(path+optim_file2):
        df_optim2 = pd.read_csv(path+optim_file2)
        fm.plotting.plot_cost_function(df_optim2, path=path2+'optimization/')
    # multistart local (instead of DE) global step, local refinement, S step of test_2sp_sepS
    for hist, hist_name in [('optimization_history_multistart.csv', '_multistart'),
                            ('optimization_history_local.csv', '_local'),
                            ('optimization_history_Smatrix.csv', '_Smatrix'),
                            ('optimization_history_Smatrix_local.csv', '_Smatrix_local')]:
        if os.path.exists(path+hist):
            fm.plotting.plot_cost_function(pd.read_csv(path+hist), path=path2+'optimization/', add_name=hist_name)

    T_x = [1. for _ in range (n_cl)]
    if RESULT_SOURCE == 'json':
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'], dtype=float)
    else:
        # Take optimal parameter values on last optimization step
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    n_param = n_cl*len(exps) + 4*n_cl + 2 + n_cl*n_cl + n_cl*n_media  # [x0 | ODE | S]
    if CASE == 'test_2sp_withS' and len(param_opt) == n_param + 1:
        param_opt = param_opt[:-1]  # result of the earlier version with s0 (last entry): drop it
    if CASE == 'test_2sp_sepS':
        # S is not part of the parameter vector: S of the S step, with which x0 + ODE were fitted
        s_x = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['s_x'], dtype=float)
        param_ode = param_opt
    else:
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
    res_real = fm.output.read_from_json(real_file, dir=data_path)
    if CASE.startswith('test_2sp'):
        # true S rows in the media order of fm.data.model_2media_expgensel, estimated S rows in sorted media order
        s_true = dict(zip(['sel1', 'gen1'], np.array(res_real['s_x'], dtype=float).reshape((n_media, n_cl))))
        print(f'S estimated / true (rows {calibr_setup["media"]}):\n', s_x, '\n',
              np.array([s_true[m] for m in calibr_setup['media']]))
    param_ode_real = np.array(res_real['param_ode'])[n_cl*len(exps):]
    x0_real = np.array(res_real['param_ode'])[:n_cl*len(exps)]

    fm.plotting.plot_parameters(param_ode, bact_all, exps, clrs1, param_real=param_ode_real, x0_real=x0_real,
                                path=path2+'optimization/')
    calibr_setup['dfs'] = data+[df_x]
    fm.plotting.plot_optimization_result(np.array(param_ode), calibr_setup, np.linspace(0, 17, 100),
                                         path=path2, clrs=clrs1, add_name=add_name+'_calibration')
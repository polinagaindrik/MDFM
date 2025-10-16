import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500

    path = 'out/'#/main_ZL2030_withS/new_result2/calibration/'
    #path2 = 'out/main_ZL2030_withS/woinhibition/'
    path2 = 'out/main_ZL2030_withS/'#new_result2/calibration/'#withinhib2_intersec/'
    add_name = '_calibr'#'_exps1_9' #'_true' #
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(data[1])[0]
    n_media = 2

    step = 1
    optim_file2 = f"optimization_history{int(step)}.csv"
    df_optim2 = pd.read_csv(path+optim_file2)

    #T_x = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
   
    T_x = [1. for _ in range(n_cl)]
    # Take optimal parameter values on last optimization step
    step = 1
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl)) 
    print(s_x)

    param_ode = param_opt[:-n_cl*n_media]
    '''
    param_ode = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    S_T = read_from_json(f'S_matrix{add_name}.json', dir=path2)
    s_x = S_T['s_x']
    T_x = S_T['T_x']
    '''
    # Plot resulting model
    calibr_setup={
        'model': fm.mdl.fusion_model2,
    #    's_x': s_x, #param_real['s_x'],#read_from_json(path2+'S_matrix_true.json'),
        'T_x': T_x, #param_real['T_x'],#[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        'output_path': path2,
        'exp_temps': fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/'),
        's_x': s_x
    }

    t_model = np.linspace(0., 17., 100)
    x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    #labels = ('day', r'log CFU mL$^{-1}$')
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_all([2., 10., 14.], labels, templ_meas=fm.plotting.plot_measurements_ZL2030_consttemp, df=data[0].filter(like=media), time_lim=[18., 10.3, 10.3],
                 temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{media}_const_model')

        
    '''
    # Plot model with all experiments together
    exps_mibi = [f'V{i:02d}' for i in range(1, 22)] + [f'V{i:02d}' for i in range(23, 29)]
    df_mibi = get_all_experiments_dataframe(read_mibi, exps_mibi, 'experiments/microbiology/')
    df_mibi = drop_column(df_mibi, ['M2', 'VRBD'])
    exps_full = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    df_mibi_constT = df_mibi.filter(regex='V.._')
    for j, media in enumerate(['MRS', 'PC']):
        plot_all([2., 10., 14.], ('day', 'log bacterian count'), templ_meas=plot_measurements_ZL2030_consttemp, df=df_mibi_constT.filter(like=media), time_lim=[18., 11., 11],
                 temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{media}_const_model1')
    '''

    data = fm.dtf.filter_dataframe_regex('_02C', data)
    data = [fm.data.drop_column(df, ['V07', 'V08', 'V16', 'V18']) for df in data]
    exps0 = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    ind_exp0 = [exps.index(i) for i in list(exps0)]
    obs_mibi_model0 = obs_mibi_model[ind_exp0]
    #ind_x0 = [i*n_cl+j for i in ind_exp0 for j in range (n_cl)]
    #x0_vals = param_ode[ind_x0]
    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_all([2.], labels, templ_meas=fm.plotting.plot_measurements_ZL2030_consttemp, df=data[0].filter(like=media), time_lim=[18.],
                 temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[ind_exp0, j], dir=path2, add_name=f'MiBi_{media}_2Grad')
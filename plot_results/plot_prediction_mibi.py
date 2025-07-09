import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500

    path_calibr = 'out/main_ZL2030_withS/new_result2/calibration/'
    add_name = '_calibr'#'_exps1_9' #'_true' #
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    df_calibr = [pd.read_pickle(path_calibr+df_name) for df_name in df_names]
    df_calibr = fm.dtf.filter_dataframe_regex('V.._', df_calibr)
    exps = sorted(list(set([s.split('_')[0] for s in df_calibr[0].columns])))
    n_cl = np.shape(df_calibr[1])[0]
    n_media = 2
    T_x = [1. for _ in range(n_cl)]
    exps_calibr = sorted(list(set([s.split('_')[0] for s in df_calibr[0].columns])))

    step = 1
    optim_file2 = f"optimization_history{int(step)}.csv"
    df_optim2 = pd.read_csv(path_calibr+optim_file2)
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl)) 
    param_model = param_opt[:-n_cl*n_media]
    param_ode = param_model[n_cl*len(exps_calibr):]
    
    path2 = 'out/main_ZL2030_withS/new_result2/prediction/x0_optim/6h_interruption/'
    predict_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        's_x': s_x,
        'output_path': path2,
        'exp_temps': fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    }
    # for 6h T-interruption
    add_name = '_prediction_6h'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    dfs_prediction = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs_prediction = [fm.output.drop_column(df, ['V10_', 'V10-CCD02']) for df in dfs_prediction]
    exps_predict = sorted(list(set([s.split('_')[0] for s in dfs_prediction[0].columns])))
    x0_opt = fm.output.read_from_json(''+'Initial_values_prediction_6h_x0optim.json', dir=path2)
    x0_vals = []
    for exp in exps_predict:
        x0_vals += x0_opt[exp]
    t_model = np.linspace(0., 17., 100)
    obs_mibi_model, obs_maldi_model, obs_ngsi_model, _ = fm.mdl.calc_obs_model(dfs_prediction, np.concatenate((x0_vals, param_ode)), predict_setup, t_model)
    #labels = ('day', r'log CFU mL$^{-1}$')
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_measurements_ZL2030_Tunterbrech(dfs_prediction[0].filter(like=media), exps_predict, mtimes=t_model,
                                             mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{media}_Tunterbrechung_6St')
    
    #    12h T-interruption
    path2 = 'out/main_ZL2030_withS/new_result2/prediction/x0_optim/12h_interruption/'
    predict_setup['output_path'] = path2
    add_name = '_prediction_12h'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    dfs_prediction = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs_prediction = [fm.output.drop_column(df, ['V10_', 'V10-CCD01']) for df in dfs_prediction]
    exps_predict = sorted(list(set([s.split('_')[0] for s in dfs_prediction[0].columns])))
    x0_opt = fm.output.read_from_json(''+'Initial_values_prediction_12h_x0optim.json', dir=path2)
    x0_vals = []
    for exp in exps_predict:
       x0_vals += x0_opt[exp]
    obs_mibi_model, obs_maldi_model, obs_ngsi_model, _ = fm.mdl.calc_obs_model(dfs_prediction, np.concatenate((x0_vals, param_ode)), predict_setup, t_model)
    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_measurements_ZL2030_Tunterbrech(dfs_prediction[0].filter(like=media), exps_predict, mtimes=t_model,
                                             mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{media}_Tunterbrechung_12St')

    dfs_prediction0 = fm.output.filter_dataframe_regex('V17', dfs_prediction)
    exps_predict0 = sorted(list(set([s.split('_')[0] for s in dfs_prediction0[0].columns])))
    x0_vals = []
    for exp in exps_predict0:
       x0_vals += x0_opt[exp]
    obs_mibi_model0, obs_maldi_model0, obs_ngsi_model0, _ = fm.mdl.calc_obs_model(dfs_prediction0, np.concatenate((x0_vals, param_ode)), predict_setup, t_model)

    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_measurements_ZL2030_Tunterbrech(dfs_prediction0[0].filter(like=media), exps_predict0, mtimes=t_model,
                                             mestim=obs_mibi_model0[:,j, : ], dir=path2, add_name=f'MiBi_{media}_Tunterbrechung_12St_V17')
    
# For x0lambda optimization:
# for 6h T-interruption
    path2 = 'out/main_ZL2030_withS/new_result2/prediction/x0lambda_optim/6h_interruption/'
    add_name = '_prediction_6h'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    dfs_prediction = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs_prediction = [fm.output.drop_column(df, ['V10_', 'V10-CCD02']) for df in dfs_prediction]
    exps_predict = sorted(list(set([s.split('_')[0] for s in dfs_prediction[0].columns])))
    optim_file = "optimization_history_predict.csv"
    df_optim = pd.read_csv(path2+optim_file)
    res_opt = df_optim.T[df_optim.T.columns[-1]].values[1:-1]
    x0_opt = res_opt[:n_cl*len(exps_predict)]
    lambd_opt = res_opt[n_cl*len(exps_predict):]


    x0_opt = fm.output.read_from_json(''+'Initial_values_prediction_6h_x0lambdaoptim.json', dir=path2)
    lambd_opt = fm.output.read_from_json(''+'Lambda_values_prediction_6h_x0lambdaoptim.json', dir=path2)
    lambd_vals, x0_vals = [], []
    for exp in exps_predict:
       x0_vals += x0_opt[exp]
       lambd_vals += lambd_opt[exp]
    res_opt = x0_vals + lambd_vals
    param_ode = param_ode[n_cl:]

    t_model = np.linspace(0., 17., 100)
    obs_mibi_model, obs_maldi_model, obs_ngs_model, _ = fm.mdl.calc_obs_model(dfs_prediction, np.concatenate((res_opt, param_ode)), predict_setup, t_model)
    #labels = ('day', r'log CFU mL$^{-1}$')
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    for j, media in enumerate(['MRS', 'PC']):
        fm.plotting.plot_measurements_ZL2030_Tunterbrech(dfs_prediction[0].filter(like=media), exps_predict, mtimes=t_model,
                                             mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{media}_Tunterbrechung_6St')
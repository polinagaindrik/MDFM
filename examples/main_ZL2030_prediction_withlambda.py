import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    path = 'out/main_ZL2030_withS/2025.06_Texponent_model/'
    workers = 16

    # Load calibration result:
    add_name_calibr = '_calibr'
    df_names = [f'dataframe_mibi{add_name_calibr}.pkl', f'dataframe_maldi{add_name_calibr}.pkl', f'dataframe_ngs{add_name_calibr}.pkl']
    dfs_calibr = [pd.read_pickle(path+'calibration/'+df_name) for df_name in df_names]
    dfs_calibr = fm.dtf.filter_dataframe_regex('V.._', dfs_calibr)
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))
  
    n_cl = np.shape(dfs_calibr[1])[0]
    T_x = [1. for _ in range(n_cl)]
    n_media = 2
    bact_all = dfs_calibr[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c

    optim_file_calibr = "optimization_history1.csv"
    df_optim_calibr = pd.read_csv(path+'calibration/'+optim_file_calibr)
    param_opt = df_optim_calibr.T[df_optim_calibr.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl)) 
    param_model = param_opt[:-n_cl*n_media]
    param_ode = param_model[n_cl*len(exps_calibr):]

    exp_temps = fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')


    # Define the experiments for the model prediction:\
    path_predict = path + 'prediction/' + 'x0lambda_optim/'
    # Constant temperature experiments (from 1 till 9) - ???

    # Experiments with 6 hours interruption
    exps_predict = ['V10', 'V12', 'V18']#, 'V20'] # no maldi data for  V20
    add_name_predict_6h = '_prediction' + '_6h'
    dfs_prediction_6h = fm.data.prepare_ZL2030_data_for_prediction(exps_predict, bact_all, 'experiments/',
                                                     path_out=path_predict+'6h_interruption/', add_name=add_name_predict_6h)
    dfs_prediction_6h = [fm.dtf.drop_column(df, ['V10_', 'V10-CCD02']) for df in dfs_prediction_6h]
    path_6h = '6h_interruption/'

    # Experiments with 12 hours interruption
    exps_predict = ['V10', 'V11', 'V17', 'V19']#, 'V20'] # no maldi data for  V20
    add_name_predict_12h = '_prediction' + '_12h'
    dfs_prediction_12h = fm.data.prepare_ZL2030_data_for_prediction(exps_predict, bact_all, 'experiments/',
                                                     path_out=path_predict+'12h_interruption/', add_name=add_name_predict_12h)
    dfs_prediction_12h = [fm.dtf.drop_column(df, ['V10_', 'V10-CCD01']) for df in dfs_prediction_12h]
    path_12h = '12h_interruption/'

    # Experiments with Lagerung
    exps_predict = ['V16'] #'V13', 'V14' are not in NGS
    add_name_predict_stor = '_prediction' + '_stor'
    dfs_prediction_stor = fm.data.prepare_ZL2030_data_for_prediction(exps_predict, bact_all, 'experiments/',
                                                      path_out=path_predict, add_name=add_name_predict_stor)

    # Experiments with industry probes  (SLH)
    exps_predict = ['V15', 'V21']#, V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    # 'V22' in different format #TODO  implement reading V22 for mibi
    # 'V23', 'V28' no MALDI
    # 'V24'-'V28' no NGS
    add_name_predict_slh = '_prediction' + '_stor'
    dfs_prediction_slh = fm.data.prepare_ZL2030_data_for_prediction(exps_predict, bact_all, 'experiments/',
                                                     path_out=path_predict, add_name=add_name_predict_slh)

    # Define the prediction setup
    x0_bnds = tuple([(1., 4.5) for _ in range (n_cl)])
    prediction_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_predict,
        'n_cl': n_cl,
        'n_media': n_media,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'param_ode': param_ode[n_cl:],
        's_x': s_x,
        'exp_temps': exp_temps,
    }

    dfs_prediction = [dfs_prediction_6h, dfs_prediction_12h]
    predict_names = [add_name_predict_6h, add_name_predict_12h]
    pathes = [path_6h, path_12h]
    for df, predict_name, path_add in zip(dfs_prediction, predict_names, pathes):
        exps_predict_full = sorted(list(set([s.split('_')[0] for s in df[0].columns])))
        print(exps_predict_full)
        x0_bnds_all = ()
        for _ in range (len(exps_predict_full)):
            x0_bnds_all+=x0_bnds
        prediction_setup['param_bnds'] = x0_bnds_all + tuple([(-6., -2.)  for _ in range (n_cl)])
        prediction_setup['exps'] = exps_predict_full
        prediction_setup['add_name'] = f'{predict_name}_x0lambdaoptim'
        prediction_setup['dfs'] = df
        
        start = time.time()
        res_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals_lambda, prediction_setup)[0]
        #print((time.time()-start)/60., 'min')
        
        #optim_file = "optimization_history_predict.csv"
        #df_optim = pd.read_csv(path_predict+path_add+optim_file)
        #fm.plotting.plot_cost_function(df_optim, path=path_predict+path_add, add_name=prediction_setup["add_name"])
        #res_opt = df_optim.T[df_optim.T.columns[-1]].values[1:-1]

        x0_opt = res_opt[:n_cl*len(prediction_setup['exps'])]
        lambd_opt = res_opt[n_cl*len(prediction_setup['exps']):]
        fm.output.save_values_each_experiment(x0_opt, prediction_setup['exps'], n_cl, dir=path_predict+path_add, filename=f'Initial_values{prediction_setup["add_name"]}')
        fm.output.save_values_each_experiment(lambd_opt, prediction_setup['exps'], n_cl, dir=path_predict+path_add, filename=f'Lambda_values{prediction_setup["add_name"]}')

        x0_opt = fm.output.read_from_json(''+f'Initial_values{prediction_setup["add_name"]}.json', dir=path_predict+path_add)
        lambda_opt = fm.output.read_from_json(''+f'Initial_values{prediction_setup["add_name"]}.json', dir=path_predict+path_add)

        fm.plotting.plot_prediction_result_x0lambda(prediction_setup['param_ode'], lambda_opt, x0_opt, prediction_setup, np.linspace(0, 17, 100),
                           path=path_predict+path_add, clrs=clrs1, add_name=prediction_setup['add_name'])
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
    path = 'model_paper/out/10_dim/'
    workers = -1

    # Load calibration result:
    add_name_calibr = ''
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

    exp_temps = fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/')


    # Define the experiments for the model prediction:
    path_predict = path + 'prediction/' + 'x0_optim/'
    exps_predict = ['V01']
    add_name_predict = '_prediction'



    dfs_prediction = fm.dtf.filter_dataframe(exps_predict[0], dfs_calibr)
    

    # Define the prediction setup
    x0_bnds = tuple([(1., 4.5) for _ in range (n_cl)])
    exps_predict_full = sorted(list(set([s.split('_')[0] for s in dfs_prediction[0].columns])))
    x0_bnds_all = ()
    for _ in range (len(exps_predict_full)):
        x0_bnds_all+=x0_bnds
    prediction_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_predict,
        'n_cl': n_cl,
        'n_media': n_media,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'param_ode': param_ode,
        's_x': s_x,
        'exp_temps': exp_temps,
        'dfs': dfs_prediction,
        'add_name': f'{add_name_predict}_x0optim',
        'param_bnds': x0_bnds_all,
        'exps': exps_predict_full
    }   
    start = time.time()
    x0_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals, prediction_setup)[0]
    print((time.time()-start)/60., 'min')
    fm.output.save_values_each_experiment(x0_opt, exps_predict_full, n_cl, dir=path_predict, filename=f'Initial_values{prediction_setup["add_name"]}')
    x0_opt = fm.output.read_from_json(''+f'Initial_values{prediction_setup["add_name"]}.json', dir=path_predict)
    fm.plotting.plot_prediction_result(prediction_setup['param_ode'], x0_opt,  prediction_setup, np.linspace(0, 17, 100),
                            path=path_predict, clrs=clrs1, add_name=prediction_setup["add_name"])
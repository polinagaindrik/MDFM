import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd


if __name__ == "__main__":
    path = 'out/'#main_ZL2030_withS/new_result2/calibration/'#'out/'
    path2 = 'out/main_ZL2030_withS/'#new_result2/calibration/'
    add_name = '_calibr'#'_exps1_9' #'_true'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    n_cl = np.shape(data[1])[0]
    n_media = 2
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    # ODE estimation:
    step = 1
    #optim_file2 = f"optimization_history{int(step)}.csv"
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2+'optimization/', #path, #
                       add_name=int(step))
                     

    T_x = [1. for _ in range (n_cl)]
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
        'exp_temps': fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    }
    calibr_setup['s_x'] = s_x
    fm.plotting.plot_parameters(param_ode, bact_all, exps, clrs1, path=path2+'optimization/')

    fm.plotting.plot_optimization_result(np.array(param_ode), calibr_setup, np.linspace(0, 17, 100),
                                    path=path2, clrs=clrs1, add_name='_calibration')
    

    '''
    #param_real = read_from_json(f'Result_calibration{add_name}.json', dir=path2)#['param_ode']
    for i, p in enumerate(param_opt):
        fig, ax = plt.subplots()
        ax.plot(df_optim2['iteration'], df_optim2[f'p{i}'])
        ax.plot(df_optim2['iteration'], p*np.ones((len(df_optim2['iteration']), 1)), color='k')
        ax.set_xlabel('iteration step', fontsize=12)
        ax.set_ylabel(f'p{i}', fontsize=12)
        #ax.set_yscale('log')
        plt.savefig(path2+'optimization/'+f'p{i}.png', bbox_inches='tight')
        plt.close(fig)
    '''
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = 'out/'
    path2 = 'out/main_ZL2030_withS/'
    add_name = '_exps1_9' #'_true'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    n_cl = np.shape(data[1])[0]
    data = fm.output.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))

    optim_file = "optimization_history_Smatrix.csv"
    df_optim = pd.read_csv(path+optim_file)
    fm.plotting.plot_cost_function(df_optim, path=path2+'optimization/', add_name='_Smatrix')

                                    # '_true'
    S_T = fm.output.read_from_json(f'S_matrix{add_name}.json', dir=path2)
    s_x = S_T['s_x']
    T_x = S_T['T_x']
    for i, si in enumerate(s_x):
        for j, sij in enumerate(si):
            fig, ax = plt.subplots()
            ax.plot(df_optim['iteration'], df_optim[f's{i}{j}'])
            ax.plot(df_optim['iteration'], sij*np.ones((len(df_optim['iteration']), 1)), color='k')
            ax.set_xlabel('iteration step', fontsize=12)
            ax.set_ylabel(f's_{i}_{j}', fontsize=12)
            plt.savefig(path2+'optimization/'+f's_{i}_{j}.png', bbox_inches='tight')
            plt.close(fig)
    s_x_last = df_optim.T[df_optim.T.columns[-1]].values[1:-2].reshape(-1, n_cl)
    s_0_last = df_optim.T[df_optim.T.columns[-1]].values[-2]
    #T_x = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    fm.plotting.plot_filteringS2(data, s_x_last, T_x, exps, path=path2)
    print(T_x)
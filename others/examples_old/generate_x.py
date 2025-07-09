import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    path = 'out/main_ZL2030_withS/new_result/'
    add_name_calibr = '_calibr'
    df_names = [f'dataframe_mibi{add_name_calibr}.pkl', f'dataframe_maldi{add_name_calibr}.pkl', f'dataframe_ngs{add_name_calibr}.pkl']
    workers = 16

    dfs_calibr = [pd.read_pickle(path+df_name) for df_name in df_names]
    dfs_calibr = fm.dtf.filter_dataframe_regex('V.._', dfs_calibr)
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))
    n_cl = np.shape(dfs_calibr[1])[0]
    n_media = 2
    bact_all = dfs_calibr[1].T.columns


    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    T_x = [1. for _ in range(n_cl)]
    # Take optimal parameter values on last optimization step

    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl)) 
    param_model = param_opt[:-n_cl*n_media]
    param_ode = param_model[n_cl*len(exps_calibr):]
    x0_vals = param_model[:n_cl*len(exps_calibr)]

    temps = [2., 10., 14.]

    # Sample x0 values from distribution (uniform distribution within param_bnds)
    L0_bnd = (1., 4.5)
    x0_bnds = tuple([(1., 4.5) for _ in range (n_cl)])
    L0_vals = np.random.uniform(*L0_bnd, size=n_cl)
    n_traj = 5
    t = np.linspace(0, 17, 17*4+1)
    for temp in temps:
        sumx_file ={'Zeit': t*24}# + [f'Gesamtkeimzahl_{j+1}' for j in range (n_traj)]
        gesamt_x = []
        for i in range (n_traj):
            L0_vals = np.random.uniform(*L0_bnd, size=n_cl)
            const = [[temp], n_cl]
            C0 = np.concatenate((10**np.array(L0_vals), np.ones((n_cl+1))))
            C = fm.mdl.model_ODE_solution(fm.mdl.fusion_model2, t, param_ode, C0, const)
            n_C = fm.mdl.get_bacterial_count(C, np.shape(s_x)[-1], 2)
            gesamt_x.append(np.sum(n_C, axis=0))
            sumx_file[f'Gesamtkeimzahl_{i+1}'] = np.sum(n_C, axis=0)
        df_res = pd.DataFrame(data=sumx_file)
        df_res.to_csv(f'out/generate_x/Gesamtkeimzahl_x_{int(temp)}Grad.csv', index=False)
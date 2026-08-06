import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

if __name__ == "__main__":
    n_cl = 4
    relnoise = 0.

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/'
    add_name = ''
    model = ode_model_coculture3

    param_opt, dfs, df_optim2 = get_param_dfs(path, path2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)

    n_exps_saved = 3
    x0_saved = param_opt[:n_cl*n_exps_saved]
    param_ode = param_opt[n_cl*n_exps_saved:]

    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    n_exps = len(names)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    param_full_sys = np.concatenate([x0_saved[:n_cl], np.zeros((n_cl)), x0_saved[n_cl:n_cl*n_exps_saved],  np.zeros((n_cl)), param_ode])
    plot_cases_separately(param_full_sys, dfs, model, path=path2, add_name=add_name, exp_indexes=[3 ,0, 2])

    #x0_vals = x0_saved param_opt[:n_cl*n_exps]
    #param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)

    # Without Ls-CTC494
    for exp in ['V02', 'V05']:
        clmns = dfs.filter(like=exp)
        dfs = dfs.drop(columns=clmns)
    names =  ['Ls23K', 'Lm', 'Ls23K-Lm']
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    for i, exp in enumerate(exps):
        data = dfs.filter(like=exp)
        plot_all_curves(param_ode_new, x0_saved[n_cl*i:n_cl*(i+1)], model=model, data=data, path=path2, add_name=f'_estim_realdata_{names[i]}')
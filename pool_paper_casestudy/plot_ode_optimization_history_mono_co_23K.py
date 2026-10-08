import os
import sys
sys.path.append(os.getcwd())
from pool_paper_casestudy import fusion_core as fm
from pool_paper_casestudy.fusion_core.mdl import ode_model_coculture_withpH_MM
from pool_paper_casestudy.fusion_core.data import get_param_dfs
from pool_paper_casestudy.fusion_core.plotting import plot_cases_separately

import numpy as np

if __name__ == "__main__":
    n_cl = 4
    relnoise = 0.

    path = 'out/'
    path2 = "pool_paper_casestudy/out/with_pH/"
    add_name = ''
    model = ode_model_coculture_withpH_MM

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
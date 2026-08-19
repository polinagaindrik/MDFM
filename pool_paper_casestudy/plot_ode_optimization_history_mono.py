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
    #path = 'pool_paper_casestudy/out/all_sepexps_withpH_Chrderiv/'
    #path2 = 'pool_paper_casestudy/out/all_sepexps_withpH_Chrderiv/'
    add_name = ''
    model = ode_model_coculture3

    _, dfs, df_optim2 = get_param_dfs(path, path2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)

    param_opt = fm.output.read_from_json('Result_calibration.json', dir=path2)["param_ode"]
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    n_exps = len(names)
    temps = [2.0 for _ in range(len(names))]
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))

    #plot_paper_figures(param_opt, dfs, path=path2, add_name=add_name)
    plot_cases_separately(param_opt, dfs, model, path=path2, add_name=add_name, exp_indexes=[0, 1, 2])
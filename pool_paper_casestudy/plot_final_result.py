import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_param_dfs(path, path2):
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    return param_opt, dfs, df_optim2


if __name__ == "__main__":
    n_cl = 4
    add_name = ''
    model = ode_model_coculture3
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    path = 'pool_paper_casestudy/out/all_together_final/'
    path2 = 'pool_paper_casestudy/out/all_together_final/'
    dfs_saved = pd.read_pickle(path2+'dataframe_poolpaper_all.pkl')

    # Monoculture experiments results:
    param_opt = fm.output.read_from_json('Result_calibration_mono_local.json', dir=path2)["param_ode"]
    plot_cases_separately(param_opt, dfs_saved, model, path=path2, add_name='_localopt_mono', exp_indexes=[0, 1, 2])

    # 3 exps (Lm+Ls23K) results:
    param_opt = fm.output.read_from_json('Result_calibration_3exps_monoco_local.json', dir=path2)["param_ode"]
    plot_cases_separately(param_opt, dfs_saved, model, path=path2, add_name='_localopt_monoco_3exps', exp_indexes=[0, 2, 3])

    # All 5 exps together 
    param_opt = fm.output.read_from_json('Result_calibration_5exps_local.json', dir=path2)["param_ode"]
    plot_cases_separately(param_opt, dfs_saved, model, path=path2, add_name='_localopt_co_5exps')
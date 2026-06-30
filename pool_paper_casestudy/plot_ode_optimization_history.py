import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd


if __name__ == "__main__":
    n_cl = 2
    n_media = 2
    relnoise = 0.1
    n_exps = 20

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/' 
    add_name = '_0'

    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)
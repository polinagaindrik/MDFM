import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def cost(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    param_ode = param#[n_cl*n_exps:]
    param_ode_new = np.copy(param_ode)
    x0_vals = calibr_setup['x0']

    (df_x, ) = calibr_setup["dfs"]
    _, [obs_x] = calibr_setup["data_array"]
    n_cl = calibr_setup['n_cl']  # np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    # TODO not clear, should just we compare logaritms?
    x_max = obs_x**2
    x_max[x_max == 0.0] = 1.0
    #ll_x = np.zeros(np.shape(obs_x))
    ll_x = np.zeros(np.shape(obs_x[:, :-1]))
    for i, exp in enumerate(exps):
        #if exp != 'LsCTC494' and exp != 'LsCTC494-Lm' and exp != 'V01' and exp != 'V05':
        if exp != 'LsCTC494-Lm' and exp != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            if calibr_setup['model'] == ode_model_coculture:
                param_ode_new[2*4 + 2 + 3+1] = 0.
            elif calibr_setup['model'] == ode_model_coculture2:
                param_ode_new[4*3+3+3] = 0.
            elif calibr_setup['model'] == ode_model_coculture3:
                param_ode_new[4*3+3+3] = 0.
            elif calibr_setup['model'] == ode_model_coculture_wopH:
                param_ode_new[3+3+3] = 0.
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode_new, x_max[i])
        else:
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return calibr_setup["aggregation_func"]([ll_x, np.min(np.concatenate([np.zeros((1, len(param))), param.reshape(1, -1)], axis=0), axis=0)**2])

def calculate_model_params(cost_func, calibr_setup):
    data_array = extract_observables_from_df(calibr_setup["dfs"])
    calibr_setup["data_array"] = data_array
    optim_output = fm.pest.optimization_func(
        cost_func,
        calibr_setup["param_bnds"],
        args=(calibr_setup, None),
        workers=calibr_setup["workers"],
    )
    return np.array(optim_output.x), optim_output.fun


if __name__ == "__main__":
    n_cl = 4
    add_name = ''

    ################ Control parameters: ######################## 
    #path = 'pool_paper_casestudy/out/all_sepexps_withpH_Chrderiv/'
    #path2 = 'pool_paper_casestudy/out/all_sepexps_withpH_Chrderiv/'
    #path = 'pool_paper_casestudy/out/all_togetherexps_withpH_Chrderiv/'
    #path2 = 'pool_paper_casestudy/out/all_togetherexps_withpH_Chrderiv/'
    #path = 'pool_paper_casestudy/out/all_3expsLmLs23K_withpH_Chrderiv/'
    #path2 = 'pool_paper_casestudy/out/all_3expsLmLs23K_withpH_Chrderiv/'
    path = 'pool_paper_casestudy/out/wo_pH_new/'
    path2 = 'pool_paper_casestudy/out/wo_pH_new/'
    add_name = '_5exps'
    model = ode_model_coculture_wopH
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']

    ## Get params from df (for interrupted exps)
    #param_opt, dfs_saved, df_optim2 = get_param_dfs(path, path2)
    #dfs = dfs_saved
    ##n_exps_saved = len(names)
    #exps_saved = sorted(list(set([s.split("_")[0] for s in dfs_saved.columns])))
    #n_exps_saved = len(exps_saved)

    ## Get params from json (for sequential exps)
    dfs_saved = pd.read_pickle(path2+'dataframe_poolpaper_all.pkl')
    param_opt = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path2)["param_ode"]
    exps_saved = sorted(list(set([s.split("_")[0] for s in dfs_saved.columns])))
    n_exps_saved = len(exps_saved)
    
    ## For monoculture exps
    #for exp in ['V04', 'V05']:
    #    clmns = dfs_saved.filter(like=exp)
    #    dfs = dfs.drop(columns=clmns)
    #names = names[:-2]

    ## For exps mono + co (Lm+Ls23K, 3 exps)
    #for exp in ['V02', 'V05']:
    #    clmns = dfs_saved.filter(like=exp)
    #    dfs = dfs.drop(columns=clmns)
    #names = ['Ls23K', 'LmCTC1034', 'Ls23K-LmCTC1034']

    # For all 5 exps: 
    dfs = dfs_saved

    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    n_exps = len(exps)

    n_exps_estim = n_exps_saved #  =n_exps only for Lm+Ls23K (old ver), for others =n_exps_saved

    data_array = extract_observables_from_df([dfs])
    x0_saved = param_opt[:n_cl*n_exps_estim]
    x0_vals = param_opt[:n_cl*n_exps]
    calibr_setup = {
            "model": model,
            "output_path": path2,
            "n_cl": n_cl,
            "dfs": [dfs],
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps,
            'data_array': data_array,
            'x0': x0_vals
    }
    param_ode = param_opt[n_cl* n_exps_estim:]
    param_loc = minimize(cost, param_ode, args=(calibr_setup, None), method='L-BFGS-B', tol=1e-8, options={'maxiter': 200, 'disp': True}).x
    #param_loc[-1] = 0. # as we set kappa_LA_lsCTC494_2 = 0 so it does not decompose LA

    #param_to_save = np.concatenate([x0_saved, param_loc]) # for all other exps
    param_to_save = np.concatenate([x0_saved[:n_cl], np.zeros((n_cl)), x0_saved[n_cl:n_cl*n_exps_saved],  np.zeros((n_cl)), param_loc]) # for Lm+Ls23K
    fm.output.json_dump({"param_ode": param_to_save.astype(list)}, f"Result_calibration{add_name}_local.json", dir=path2)
    print("Locally optimized parameters:", param_loc)

    x0_vals = calibr_setup['x0']
    param_ode = list(param_loc)
    param_ode_new = np.copy(param_ode)
    plot_cases_separately(np.concatenate([x0_saved, param_ode]) , dfs_saved, model, path=path2, add_name=f'{add_name}_localopt')
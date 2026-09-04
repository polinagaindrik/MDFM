import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def multistart_minimize(cost_func, param_ode_init, calibr_setup, n_restarts=6, jitter_frac=0.15,
                         biological_guess=None, seed=None, method='L-BFGS-B', tol=1e-8, maxiter=200):
    """
    Run L-BFGS-B from multiple starting points, keep the best result.

    param_ode_init   : your current best-guess param_ode vector (e.g. from a prior fit)
    n_restarts       : number of restarts total, INCLUDING the run from param_ode_init itself
    jitter_frac      : relative random perturbation applied to param_ode_init for restarts
                        that aren't the biological_guess or the unperturbed start
    biological_guess : optional dict {index: value} to override specific entries of
                        param_ode_init for one of the restarts (e.g. {3: 1.6, 4: 2800.}
                        for MM's omega3, K3) -- guarantees that specific hypothesis gets tested
    seed              : int, for reproducible jitter across restarts
    """
    rng = np.random.default_rng(seed)
    bnds = calibr_setup["param_bnds"]
    lo_arr = np.array([b[0] for b in bnds])
    hi_arr = np.array([b[1] for b in bnds])
    span = hi_arr - lo_arr

    starts = []
    starts.append(("unperturbed", np.array(param_ode_init, dtype=float)))

    if biological_guess is not None:
        bio_start = np.array(param_ode_init, dtype=float)
        for idx, val in biological_guess.items():
            bio_start[idx] = val
        bio_start = np.clip(bio_start, lo_arr, hi_arr)
        starts.append(("biological_guess", bio_start))

    n_random = max(0, n_restarts - len(starts))
    for r in range(n_random):
        jitter = rng.normal(0.0, jitter_frac, size=len(param_ode_init)) * np.where(span > 0, span, 1.0)
        jittered = np.clip(np.array(param_ode_init, dtype=float) + jitter, lo_arr, hi_arr)
        starts.append((f"random_{r}", jittered))

    best_res, best_label = None, None
    for label, x0 in starts:
        res = minimize(cost_func, x0, args=(calibr_setup, None), method=method,
                        tol=tol, bounds=bnds, options={'maxiter': maxiter})
        print(f"[{label}] success={res.success}, nit={res.nit}, cost={res.fun:.6g}")
        if best_res is None or res.fun < best_res.fun:
            best_res, best_label = res, label

    print(f"\nBest start: '{best_label}', cost={best_res.fun:.6g}")
    return best_res


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
            elif calibr_setup['model'] == ode_model_coculture_wopH_MM:
                param_ode_new[8] = 0.
            elif calibr_setup['model'] == ode_model_coculture_wopH_expsat:
                param_ode_new[8] = 0.
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
    #path = 'pool_paper_casestudy/out/wo_pH_new/'
    #path2 = 'pool_paper_casestudy/out/wo_pH_new/'
    path = 'pool_paper_casestudy/out/wo_pH_new/'
    path2 = 'pool_paper_casestudy/out/wo_pH_new/'
    add_name = '_5exps_MM'
    model = ode_model_coculture_wopH_MM
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']

    ### Get params from df (for interrupted exps)
    ##param_opt, dfs_saved, df_optim2 = get_param_dfs(path, path2)

    ## Get params from json (for sequential exps)
    dfs_saved = pd.read_pickle(path2+'dataframe_poolpaper_all.pkl')
    param_opt = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path2)["param_ode"]

    ## For all setups: 
    dfs = dfs_saved
    #n_exps_saved = len(names)
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


    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    n_exps = len(exps)

    n_exps_estim = n_exps_saved #  =n_exps only for Lm+Ls23K (old ver), for others =n_exps_saved

    data_array = extract_observables_from_df([dfs])
    x0_saved = param_opt[:n_cl*n_exps_estim]
    x0_vals = param_opt[:n_cl*n_exps]
    #x0_vals = np.concatenate([param_opt[:n_cl], param_opt[n_cl*2:n_cl*4]]) # only for Lm-Ls23K setup
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
    param_ode_bnds = tuple(
            [(.2, 1.), (.2, 1.), (.2, 1.)] + # mu_opt
            #[(0.9, 1.2), (700., 14000.), (0.25, 0.5)] + # omegaT_exp + ki_T_inhib + n 
            [(0.05, 3.0), (1, 1000)] + 
            [(8.0, 9.0), (8., 9.), (8., 9.0)]  + # N_max_exp
            [(.2, 1.)] + # kappa_T
            [(0.2, 1)] + [(2., 10)] +   # kappa_LA ls23K
            [(0.1, 1.)] + [(3.5, 10)] +   # kappa_LA lsCTC494
            [(0., 0.)] + [(0., 2.)]     # kappa_LA lm
        )
    calibr_setup["param_bnds"] = param_ode_bnds
    param_ode = param_opt[n_cl* n_exps_estim:]
    # indices of omega3, K3 in the 15-length MM param_ode vector

    res = multistart_minimize(
        cost, param_ode, calibr_setup,
        n_restarts=5, jitter_frac=0.15,
        seed=0,
    )
    param_loc = res.x
    #param_loc[-1] = 0. # as we set kappa_LA_lsCTC494_2 = 0 so it does not decompose LA

    param_to_save = np.concatenate([x0_saved, param_loc]) # for all other exps
    #param_to_save = np.concatenate([x0_saved[:n_cl], np.zeros((n_cl)), x0_saved[n_cl:n_cl*n_exps_saved],  np.zeros((n_cl)), param_loc]) # for Lm+Ls23K (old ver)
    fm.output.json_dump({"param_ode": param_to_save.astype(list)}, f"Result_calibration{add_name}_local.json", dir=path2)

    param_ode = list(param_loc)
    plot_cases_separately(param_to_save , dfs_saved, model, path=path2, add_name=f'{add_name}_localopt')
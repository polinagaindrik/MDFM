import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_calibration_poolpaper_sequen(dfs, path=""):
    n_cl = 4
    model = ode_model_coculture3
    dicts_param = {}
    x0_vals = {}
    # Monoculture experiments
    for i in [2, 0, 1]:
        df = [dfs[0].filter(like=f'V{i+1:02d}')]#.filter(like=f'V{i+1:02d}')
        exps_calibr = sorted(list(set([s.split("_")[0] for s in df[0].columns])))
        calibr_presetup = {
            "model": model,
            "workers": workers,  # number of threads for multiprocessing
            "output_path": path,
            "n_cl": n_cl,
            "dfs": df,
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps_calibr,
        }
        x0_bnds_all = []
        for exp in calibr_presetup["exps"]:
            if exp == 'V01' or exp == 'V04': # ls23K
                add = [(df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.2*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
                df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.2*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper']), (0., 0.)]
            else: # ls494 
                add = [(0., 0.),
                (df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.5*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
                df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.5*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'])]
            #and lm
            add += [(df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] - 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'], df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] + 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # lm_sens
            if exp == 'V05':
                add += [(0., 0.1*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper']) # with resistant bacteria
                ]
            else: 
                add += [(0., 0.)] # with resistant bacteria
            x0_bnds_all += add
        x0_bnds_all = tuple(x0_bnds_all)

        if i == 0:
            param_ode_bnds_mono = tuple(
                [(.2, 1.), (0., 0.), (0., 0.)] + # mu_opt
                [(1., 3.5),(5., 8.), (9., 14.),
                (3., 3.), (7., 7.), (9., 9.),
                (3., 3.), (7., 7.), (9., 9.),] +  # pH_min, pH_opt, pH_max
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
                [(8., 9.5), (0., 0.), (dicts_param[2]['N_t'], dicts_param[2]['N_t'])] + # N_max_exp
                [(0., 0.)] + # kappa_T
                [(.1, 10)] + [(1., 100.)] + # kappa_LA ls23K
                [(0., 0.)] + [(0., 0.)] + # kappa_LA lsCTC494
                [(0., 0.)] + [(0., 0.)] # kappa_LA lm
            )
        elif i == 1:
            param_ode_bnds_mono = tuple(
                [(0., 0.), (.2, 1.), (0., 0.)] + # mu_opt
                [(3., 3.), (7., 7.), (9., 9.),
                (1., 3.5), (5., 8.), (9., 14.),
                (3., 3.), (7., 7.), (9., 9.),] +  # pH_min, pH_opt, pH_max
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
                [(dicts_param[0]['N_t'], dicts_param[0]['N_t']), (8., 9.5), (dicts_param[2]['N_t'], dicts_param[2]['N_t'])] + # N_max_exp
                [(0., 0.)] + # kappa_T
                [(0., 0.)] + [(0., 0.)] + # kappa_LA ls23K
                [(.1, 10)] + [(1., 100.)] + # kappa_LA lsCTC494
                [(0., 0.)] + [(0., 0.)] # kappa_LA lm
            )
        elif i == 2:
            param_ode_bnds_mono = tuple(
                [(0., 0.), (0., 0.), (.2, 1.)] + # mu_opt
                [(3., 3.), (7., 7.), (9., 9.),
                (3., 3.), (7., 7.),  (9., 9.),
                (1., 3.5), (6., 8.), (9., 14.)] +  # pH_min, pH_opt, pH_max
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n  
                [(0., 0.), (0., 0.), (8., 9.5)]  +# rj, N_max_exp
                [(0., 0.)] + # kappa_T
                [(0., 0.)] + [(0., 0.)] +   # kappa_LA ls23K
                [(0., 0.)] + [(0., 0.)] +   # kappa_LA lsCTC494
                [(.1, 10)] + [(1., 100.)]   # kappa_LA lm
            )
        calibr_setup = calibr_presetup
        calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds_mono
        print(f"Start optimization {i}")
        param_opt = calculate_model_params(cost, calibr_setup)[0]
        x0_vals[i] = param_opt[:n_cl]
        dicts_param[i] = get_extract_params_from_mono_exp(param_opt[n_cl:], i)

        df_optim2 = pd.read_csv('out/optimization_history1.csv')
        fm.plotting.plot_cost_function(df_optim2, path=path, add_name=f'_V{i+1:02d}')

    # Estimation of x0 for coculture ls23K and Lm
    df = [dfs[0].filter(like='V04')]
    exps_calibr = sorted(list(set([s.split("_")[0] for s in df[0].columns])))
    calibr_presetup = {
            "model": model,
            "workers": workers,  # number of threads for multiprocessing
            "output_path": path,
            "n_cl": n_cl,
            "dfs": df,
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps_calibr,
        }
    x0_bnds_all = []
    for exp in calibr_presetup["exps"]:
        add = [(df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.2*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
        df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.2*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper']), (0., 0.)]
        #and lm
        add += [(df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] - 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'], df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] + 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # lm_sens
        add += [(0., 0.)] # with resistant bacteria
        x0_bnds_all += add
    x0_bnds_all = tuple(x0_bnds_all)    
    param_ode_bnds = tuple(
        [(dicts_param[j]['mu_opt'], dicts_param[j]['mu_opt']) for j in range (3)] + [
        (dicts_param[0]['pH_min'], dicts_param[0]['pH_min']), (dicts_param[0]['pH_opt'], dicts_param[0]['pH_opt']), (dicts_param[0]['pH_max'], dicts_param[0]['pH_max']),
        (dicts_param[1]['pH_min'], dicts_param[1]['pH_min']), (dicts_param[1]['pH_opt'], dicts_param[1]['pH_opt']), (dicts_param[1]['pH_max'], dicts_param[1]['pH_max']),
        (dicts_param[2]['pH_min'], dicts_param[2]['pH_min']), (dicts_param[2]['pH_opt'], dicts_param[2]['pH_opt']), (dicts_param[2]['pH_max'], dicts_param[2]['pH_max'])] +
        [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
        [(dicts_param[j]['N_t'], dicts_param[j]['N_t']) for j in range (3)] +
        [(0., 0.)] + # kappa_T
        [(kappa, kappa)  for j in range (3) for kappa in dicts_param[j]['kappas_LA']]
        )

    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds
    print("Start optimization Ls23K-Lm")
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    x0_vals[3] = param_opt[:n_cl]
    df_optim2 = pd.read_csv('out/optimization_history1.csv')
    fm.plotting.plot_cost_function(df_optim2, path=path, add_name=f'_V04')

####################################################
    
    # Now estimate the rest parameters for coculture lsCTC494 and Lm
    df = [dfs[0].filter(like='V05')]
    exps_calibr = sorted(list(set([s.split("_")[0] for s in df[0].columns])))
    calibr_presetup = {
            "model": model,
            "workers": workers,  # number of threads for multiprocessing
            "output_path": path,
            "n_cl": n_cl,
            "dfs": df,
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps_calibr,
        }
    x0_bnds_all = []
    for exp in calibr_presetup["exps"]:
        add = [(0., 0.),
        (df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.5*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
        df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.5*df[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'])]
        #and lm
        add += [(df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] - 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'], df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] + 0.3*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # lm_sens
        add += [(0., 0.1*df[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # with resistant bacteria
        x0_bnds_all += add
    x0_bnds_all = tuple(x0_bnds_all)
    param_ode_bnds = tuple(
         [(dicts_param[j]['mu_opt'], dicts_param[j]['mu_opt']) for j in range (3)] + [
        (dicts_param[0]['pH_min'], dicts_param[0]['pH_min']), (dicts_param[0]['pH_opt'], dicts_param[0]['pH_opt']), (dicts_param[0]['pH_max'], dicts_param[0]['pH_max']),
        (dicts_param[1]['pH_min'], dicts_param[1]['pH_min']), (dicts_param[1]['pH_opt'], dicts_param[1]['pH_opt']), (dicts_param[1]['pH_max'], dicts_param[1]['pH_max']),
        (dicts_param[2]['pH_min'], dicts_param[2]['pH_min']), (dicts_param[2]['pH_opt'], dicts_param[2]['pH_opt']), (dicts_param[2]['pH_max'], dicts_param[2]['pH_max'])] +
        [(0.5, 2.), (3000., 5000.), (0.3, 1.5)] + # omegaT_exp + ki_T_inhib + n
        [(dicts_param[j]['N_t'], dicts_param[j]['N_t']) for j in range (3)] +
        [(.1, 1.)]  + # kappa_T
        [(kappa, kappa)  for j in range (3) for kappa in dicts_param[j]['kappas_LA']]
        )
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds
    print("Start optimization  LsCTC494-Lm")
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    x0_vals[4] = param_opt[:n_cl]
    x0_vals_final = np.array([x0_vals[i] for i in range (5)]).flatten()
    param_final = np.concatenate((x0_vals_final, param_opt[n_cl:]))
    fm.output.json_dump({"param_ode": param_final.astype(list)}, "Result_calibration.json", dir=path)
    df_optim2 = pd.read_csv('out/optimization_history1.csv')
    fm.plotting.plot_cost_function(df_optim2, path=path, add_name=f'_V05')
    return param_final, calibr_setup

def get_extract_params_from_mono_exp(param_ode, i):
    n_cl_param = 3
    mu_opt = param_ode[i]
    pH_params = param_ode[n_cl_param + i*3:n_cl_param + i*3+3]
    N_t = param_ode[4*n_cl_param +3 + i]
    kappas_LA = param_ode[5*n_cl_param +3+1+ 2*i: 5*n_cl_param +3+1 + (2*i+2)]
    dict_exp = {'mu_opt': mu_opt, 'pH_min': pH_params[0], 'pH_opt': pH_params[1], 'pH_max': pH_params[2], 'N_t': N_t, 'kappas_LA': kappas_LA}
    print(i, dict_exp)    
    return dict_exp


if __name__ == "__main__":
    path = "pool_paper_casestudy/out/"
    workers = -1
    n_cl = 4
    model = ode_model_coculture3
    path_new = path + "test2/"
    
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    skip_rows = [34, 8,  58, 109, 83]
    LA_sheetnames = ['R9_23K_LA_prod', 'R9_494_LA_prod', 'R9_1034_LA_prod', 'R9_23Kco_LA_prod', 'R9_494co_LA_prod']

    df_exps = []
    for i, (n, nr, las) in enumerate(zip(names, skip_rows, LA_sheetnames)):
        df_exps.append(experimental_values(n, skiprows=nr, path_data='pool_paper_casestudy/data/', LA_sheetname=las, path=path_new, exp_start_offset=i))
    dfs = fm.dtf.merge_dfs(df_exps, sort=False)
    fm.data.save_all_dfs([dfs], names=['poolpaper_all'], path=path_new)

    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    n_exps = len(exps)

    param_opt, calibr_setup = data_calibration_poolpaper_sequen([dfs], path=path_new)

    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    for i in range (len(exps)):
        data = dfs.filter(like=f'V{i+1:02d}')
        #if exp != 'LsCTC494' and exp != 'LsCTC494-Lm' and exp != 'V01' and exp != 'V05':
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode_new[4*3+3+3] = 0.
            plot_all_curves(param_ode_new, x0_vals[n_cl*i:n_cl*(i+1)], model=model, data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')
        else:
            plot_all_curves(param_ode, x0_vals[n_cl*i:n_cl*(i+1)], model=model, data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')
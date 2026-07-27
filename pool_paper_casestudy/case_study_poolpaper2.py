import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ode_model_coculture2(t, x, param, x0, ode_args):
    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    (x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_lsCTC494_min, pH_lsCTC494_opt, pH_lm_min, pH_lm_opt,
    omegaT_lm, k_T_inhib, n,
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    mu_ls = mu_ls_opt * (pH - pH_ls_min) / (pH_ls_opt - pH_ls_min)
    mu_lm = mu_lm_opt * (pH - pH_lm_min) / (pH_lm_opt - pH_lm_min)

    N_ls23K_t, N_lsCTC494_t, N_lm_t = 10**np.array([N_ls23K_texp, N_lsCTC494_texp, N_lm_texp])
    kappa_T = 10**(-5) * kappa_T_0


    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    toxin_death = omegaT_lm * x_lm_sen * T**n / (k_T_inhib**n + T**n)

    return [
        (mu_ls * R) * x_ls23K,
        (mu_ls * R) * x_lsCTC494,
        (mu_lm * R) * x_lm_sen - toxin_death,
        (mu_lm * R) * x_lm_res,
        -(mu_ls / N_t)*R*x_ls23K - (mu_ls / N_t)*R*x_lsCTC494 - (mu_lm / N_t)*R*x_lm_sen - (mu_lm / N_t)*R*x_lm_res,
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]

def data_calibration_poolpaper_sequen(dfs, path=""):
    dicts_param_ode = []
    # Monoculture experiments
    for i in range (3):
        df = dfs.filter(like=f'V{i+1:02d}')
        exps_calibr = sorted(list(set([s.split("_")[0] for s in dfs[0].columns])))
        calibr_presetup = {
            "model": ode_model_coculture,
            "workers": workers,  # number of threads for multiprocessing
            "output_path": path,
            "n_cl": n_cl,
            "dfs": df,
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps_calibr,
        }
        x0_bnds_all = []
        for exp in calibr_presetup["exps"]:
            if exp == 'V02' or exp == 'V04': # ls23K
                add = [(dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.2*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
                dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.2*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper']), (0., 0.)]
            else: # ls494 
                add = [(0., 0.),
                (dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.5*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
                dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.5*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'])]
            #and lm
            add += [(dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] - 0.3*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'], dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] + 0.3*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # lm_sens
            if exp == 'V05':
                add += [(0., 0.1*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper']) # with resistant bacteria
                ]
            else: 
                add += [(0., 0.)] # with resistant bacteria
            x0_bnds_all += add
        x0_bnds_all = tuple(x0_bnds_all)

        if i == 0:
            param_ode_bnds_mono = tuple(
                [(.2, .7), (0., 0.), (0., 0.)] + # mu_opt
                [(3.5, 4.5), (5., 8.), (3., 3.), (7., 7.), (3., 3.), (7., 7.),] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
                [(8.0, 9.5), (1., 1.), (1., 1.)] +          # N_max_exp
                [(0., 0.)]  + # kappa_T
                [(.1, 10)] + [(1., 100.)] + # kappa_LA ls23K
                [(0., 0.)] + [(0., 0.)] + # kappa_LA lsCTC494
                [(0., 0.)] + [(0., 0.)] # kappa_LA lm
            )
        elif i == 1:
            param_ode_bnds_mono = tuple(
                [(0., 0.), (.2, .7), (0., 0.)] + # mu_opt
                [(3., 3.), (7., 7.), (3.5, 4.5), (5., 8.), (3., 3.), (7., 7.),] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
                [(1., 1.), (8.0, 9.5), (1., 1.)] +          # N_max_exp
                [(0., 0.)]  + # kappa_T
                [(0., 0.)] + [(0., 0.)] + # kappa_LA ls23K
                [(.1, 10)] + [(1., 100.)] + # kappa_LA lsCTC494
                [(0., 0.)] + [(0., 0.)] # kappa_LA lm
            )
        elif i == 2:
            param_ode_bnds_mono = tuple(
                [(.2, .7), (0., 0.), (0., 0.)] + # mu_opt
                [(3., 3.), (7., 7.), (3., 3.), (7., 7.), (3.5, 4.5), (5., 8.)] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
                [(0., 0.)] + [(1., 1.)] + [(1, 1)] + # omegaT_exp + ki_T_inhib + n
                [(1., 1.), (1., 1.), (8.0, 9.5)] +   # N_max_exp
                [(0., 0.)] + # kappa_T
                [(0., 0.)] + [(0., 0.)] +   # kappa_LA ls23K
                [(0., 0.)] + [(0., 0.)] +   # kappa_LA lsCTC494
                [(.1, 10)] + [(1., 100.)]   # kappa_LA lm
            )
        calibr_setup = calibr_presetup
        calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds_mono
        print("Start optimization...")
        param_opt = calculate_model_params(cost, calibr_setup)[0]
        dicts_param.append(get_extract_params_from_mono_exp(param_opt, i))
    param_param_ode = (
        [(dicts_param[j]['mu_opt'], dicts_param[j]['mu_opt']) for j in range (3)] + [
        (dicts_param[0]['pH_min'], dicts_param[0]['pH_min']), (dicts_param[0]['pH_opt'], dicts_param[0]['pH_opt']),
        (dicts_param[1]['pH_min'], dicts_param[1]['pH_min']), (dicts_param[1]['pH_opt'], dicts_param[1]['pH_opt']),
        (dicts_param[2]['pH_min'], dicts_param[2]['pH_min']), (dicts_param[2]['pH_opt'], dicts_param[2]['pH_opt'])] +
        [(0.5, 2.), (3000., 5000.), (0.3, 1.5)] + # omegaT_exp + ki_T_inhib + n
        [(dicts_param[j]['N_t'], dicts_param[j]['N_t']) for j in range (3)] +
        [(.1, 1.)]  + # kappa_T
        [(.1, 10)] + [(1., 100.)] + # kappa_LA ls23K
        [(.1, 10)] + [(1., 100.)] + # kappa_LA lsCTC494
        [(.1, 10)] + [(1., 100.)] # kappa_LA lm
        )
    return 1

def get_extract_params_from_mono_exp(param_opt, i):
    n_cl_param = 3
    mu_opt = param_opt[i]
    pH_min = param_opt[n_cl_param + i]
    pH_opt = param_opt[2*n_cl_param + i]
    N_t = param_opt[2*n_cl_param + 3 + i]
    return {'mu_opt': mu_opt, 'pH_min': pH_min, 'pH_opt': pH_opt, 'N_t': N_t}

def data_calibration_poolpaper(dfs, path=""):
    exps_calibr = sorted(list(set([s.split("_")[0] for s in dfs[0].columns])))
    calibr_presetup = {
        "model": ode_model_coculture,
        "workers": workers,  # number of threads for multiprocessing
        "output_path": path,
        "n_cl": n_cl,
        "dfs": dfs,
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps_calibr,
    }
    x0_bnds_all = tuple([(2., 6.) for _ in range(calibr_presetup["n_cl"])])

    x0_bnds_all = []
    for exp in calibr_presetup["exps"]:
        if exp == 'V02' or exp == 'V04': # ls23K
            add = [(dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.2*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
            dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.2*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper']), (0., 0.)]
        else: # ls494 
            add = [(0., 0.),
            (dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] - 0.5*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'],
            dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'] + 0.5*dfs[0].T['x_Ls_State_00'][f'{exp}_01_poolpaper'])]
        #and lm
        add += [(dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] - 0.3*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'], dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'] + 0.3*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper'])] # lm_sens
        if exp == 'V05':
            add += [(0., 0.1*dfs[0].T['x_Lm_State_00'][f'{exp}_01_poolpaper']) # with resistant bacteria
            ]
        else: 
            add += [(0., 0.)] # with resistant bacteria
        x0_bnds_all += add
    x0_bnds_all = tuple(x0_bnds_all)

    param_ode_bnds = tuple(
        [(.2, .7), (.2, .7)] + # mu_opt
        #[(0.55, 0.55), (0.33, 0.33)] + # mu_opt
        [(3.83, 3.83), (7.02, 7.02), (4.84, 4.84), (5.81, 5.81)] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
        #[(3.5, 4.5), (5., 8.), (3.5, 5.), (5.5, 8.)] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
        #[(0.1, 10.) for _ in range(2)] +  # omega
        [(0., 0.) for _ in range(2)] +  # omega
        [(0.5, 2.)] + [(3000., 5000.)] + [(0.3, 1.5)] + # omegaT_exp + ki_T_inhib + n
        [(8.0, 9.5)] +          # N_max_exp
        #[(8.2855, 8.2855)] +          # N_max_exp
        [(.1, 1.)]  + # kappa_T
        #[(0.6, 0.6)]  + # kappa_T
        #[(0., 0.)] + # kappa_T
        [(-11, -9)] + [(-9.5, -8.)] + # kappa_LA ls23K
        [(-11, -9)] + [(-9.5, -8.)] + # kappa_LA lsCTC494
        [(-10., -9.)] + # kappa_LA lm
        [(0., 0.)] # q_acid
        #[(10**(-3), 6*10**(-3))] # q_acid
    )
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds

    print("Start optimization...")
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    fm.output.json_dump({"param_ode": param_opt.astype(list)}, "Result_calibration.json", dir=path)
    return param_opt, calibr_setup


if __name__ == "__main__":
    path = "pool_paper_casestudy/out/"
    workers = -1
    n_cl = 4
    n_states = 1
    # relnoise = 0.1
    add_name = ""
    ntr = 1
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

    # Test pH(LA) dependence
    #days, [obs_x] = extract_observables_from_df([df_exps[3]])
    #pH_LA_dependence(days, obs_x[0][4], obs_x[0][5], add_name='_test_direct_calulation', path=path_new)

    param_opt, calibr_setup = data_calibration_poolpaper([dfs], path=path_new)

    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    for i in range (len(exps)):
        data = dfs.filter(like=f'V{i+1:02d}')
        #if exp != 'LsCTC494' and exp != 'LsCTC494-Lm' and exp != 'V01' and exp != 'V05':
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode_new[2*3 + 2 + 3+1] = 0.
            plot_all_curves(param_ode_new, x0_vals[n_cl*i:n_cl*(i+1)], data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')
        else:
            plot_all_curves(param_ode, x0_vals[n_cl*i:n_cl*(i+1)], data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')    
    
    '''
    # Test with in-siilico data generation and calibration
    ## Test the model results:
    t_test = np.linspace(0.0, 55.0, 100)  # hours
    x0_test = np.array([10**5., 10**3.1, 1.0, 0.0, 0.0, 6.0])
    param_ode_test = [
                        0.38,       # mu_ls
                        0.32,       # mu_lm
                        -3,         # omega_ls
                        -3,         # omega_lm
                        4.,         # omegaT_lm
                        8.3,        # N_texp
                        -5,         # k_T
                        -9,         # k_LA_ls
                        -9.1,       # k_LA_lm
                     ]
    x10_bact = [[5., 3.1]]
    plot_all_curves(t_test, param_ode_test, x10_bact, path=path_new, add_name='_init')

    df_ode = data_generation_poolpaper(n_cl, param_ode_test, x10_bact, t_test, path=path_new)
    days_x, [obs_x] = extract_observables_from_df([df_ode])
    param_opt, calibr_setup = data_calibration_poolpaper([df_ode], path=path_new)
    plot_all_curves(t_test, param_opt[n_cl*len(temps):], [param_opt[:n_cl*len(temps)]], path=path_new, add_name='_estim')
    '''     
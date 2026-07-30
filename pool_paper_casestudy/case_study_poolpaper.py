import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_calibration_poolpaper(dfs, path=""):
    exps_calibr = sorted(list(set([s.split("_")[0] for s in dfs[0].columns])))
    calibr_presetup = {
        "model": ode_model_coculture2, #ode_model_coculture,
        "workers": workers,  # number of threads for multiprocessing
        "output_path": path,
        "n_cl": n_cl,
        "dfs": dfs,
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps_calibr,
    }
    x0_bnds_all = []
    for exp in calibr_presetup["exps"]:
        if exp == 'V01' or exp == 'V04': # ls23K
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
    '''
    param_ode_bnds = tuple(
        [(.2, .7), (.2, .7)] + # mu_opt
        #[(0.55, 0.55), (0.33, 0.33)] + # mu_opt
        #[(3.83, 3.83), (7.02, 7.02), (4.84, 4.84), (5.81, 5.81)] +  # pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt
        [(1., 3.5), (6., 9.), (9., 14.),      # pH_ls_min, pH_ls_opt, pH_ls_max
         (1., 3.5), (6., 8.), (9., 14.)] +    # pH_lm_min, pH_lm_opt, pH_lm_max
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
    '''
    param_ode_bnds = tuple(
            [(.2, 1.) for _ in range (3)] + # mu_opt
            [(1., 3.5), (6., 8.), (9., 14.),
            (1., 3.5), (6., 8.), (9., 14.),
            (1., 3.5), (6., 8.), (9., 14.)] +  # pH_min, pH_opt, pH_max
            [(0.5, 2.), (3000., 5000.), (0.3, 1.5)] + # omegaT_exp + ki_T_inhib + n  
            [(1., 5.), (1., 5.), (8., 9.)]  +# rj, N_max_exp
            [(.1, 1.)] + # kappa_T
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA ls23K
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA lsCTC494
            [(.1, 10)] + [(1., 100.)]     # kappa_LA lm
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
    path_new = path + "test/"
    
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
            param_ode_new[2*4 + 2 + 3+1] = 0.
            plot_all_curves(param_ode_new, x0_vals[n_cl*i:n_cl*(i+1)], data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')
        else:
            plot_all_curves(param_ode, x0_vals[n_cl*i:n_cl*(i+1)], data=data, path=path_new, add_name=f'_estim_realdata_{names[i]}')
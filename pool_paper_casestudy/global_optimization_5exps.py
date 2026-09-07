import os
import sys

sys.path.append(os.getcwd())
from pool_paper_casestudy import fusion_core as fm
from pool_paper_casestudy.fusion_core.mdl import cost, ode_model_coculture_withpH_MM
from pool_paper_casestudy.fusion_core.pest import calculate_model_params
from pool_paper_casestudy.fusion_core.data import experimental_values


def data_calibration_poolpaper(dfs, path="", add_name="", n_cl=4):
    exps_calibr = sorted(list(set([s.split("_")[0] for s in dfs[0].columns])))
    model = ode_model_coculture_withpH_MM
    calibr_presetup = {
        "model": model, #ode_model_coculture,
        "workers": workers,  # number of threads for multiprocessing
        "add_name": add_name,
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
    param_ode_bnds = tuple(
            [(.2, 1.) for _ in range (3)] + # mu_opt
            [(1., 3.5), (6., 8.), (9., 14.),
             (1., 3.5), (6., 8.), (9., 14.),
             (1., 3.5), (6., 8.), (9., 14.)] +  # pH_min, pH_opt, pH_max
            [(0.05, 3.0), (1, 1000)] +  # omega, K_m
            [(.1, 1.)] + # kappa_T
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA ls23K
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA lsCTC494
            [(0., 0.)] + [(1., 100.)]     # kappa_LA lm
        )  
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds
    print("Start optimization...")
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    fm.output.json_dump({"param_ode": param_opt.astype(list)}, f"Result_calibration{add_name}.json", dir=path)
    return param_opt, calibr_setup


if __name__ == "__main__":
    workers = -1
    n_cl = 4
    model = ode_model_coculture_withpH_MM # ode_model_coculture_withoutpH_MM
    path_new = "pool_paper_casestudy/out/with_pH/" # wo_pH/
    add_name = "_5exps_MM_withpH"
    
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
    param_opt, calibr_setup = data_calibration_poolpaper([dfs], path=path_new, add_name=add_name, n_cl=n_cl)
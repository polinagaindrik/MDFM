import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 


def plot_cases_separately(param_opt, dfs, model, path='', add_name=''):
    names = ['Ls23K', 'LmCTC1034', 'Ls23K-LmCTC1034']
    n_exps = len(names)
    coord_text = (0.04, 0.88)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    #param_ode_new[2*4 + 2 + 3+1] = 0. #model1
    param_ode_new[4*3+3+3] = 0. # model2

    days, [obs_x] = extract_observables_from_df([dfs])
    t_model = np.linspace(days[0], days[-1]+5, 100)

    exp_indexes = [2, 0, 1]
    lbls = ['Ls-23K', 'Ls-CTC494',  'Lm-CTC1034', 'Lactic Acid', 'pH']
    mrkrs = ['o', 'o',  'o', '^', 'x']
    lst = ['solid', 'solid', 'solid', 'dashed', '']
    clrs = [colors_all['N_LsCTC494'], colors_all['N_LsCTC494co'], colors_all['N_Lm_withT'], colors_all['T_A'], colors_all['N_Ls23K']]
    clr_indexes = [[0, 3, 4], [2, 3, 4], [0, 2, 3, 4]]
    obs_count_indexes = [[0], [1], [0, 1]]
    subfigures = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']#, r'\textbf{D}', r'\textbf{E}']

    for i in exp_indexes:
        index = clr_indexes[i]
        clrs_exp = [clrs[ind] for ind in index]
        lbls_exp = [lbls[ind] for ind in index]
        mrkrs_exp = [mrkrs[ind] for ind in index]
        lst_exp = [lst[ind] for ind in index]
        obs_count_ind_exp = obs_count_indexes[i]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)

        for k in range(len(index)-2):
            ax.plot(t_model, obs_model[obs_count_ind_exp[k]], color=clrs_exp[k], linewidth=3, linestyle=lst_exp[k])
            ax.scatter(days, obs_x[i][obs_count_ind_exp[k]], marker=mrkrs_exp[k], color=clrs_exp[k])
        
        ax2.plot(t_model, obs_model[3], linewidth=3, color=clrs_exp[k+1], linestyle=lst_exp[k+1])
        ax2.scatter(days, obs_x[i][3], color=clrs_exp[k+1], marker=mrkrs_exp[k+1])
        ax2.scatter(days, obs_x[i][4], color=clrs_exp[k+2], marker=mrkrs_exp[k+2])

        ax.set_xlim(-0.05, np.max(t_model))
        ax.set_yscale('log')
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
        fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'pH; Lactic Acid [g/L]')
        ax2.set_ylim(-0.5, 7)
        legend_elements = [
            Line2D([0], [0], color=clrs_exp[j], label=lbls_exp[j], marker=mrkrs_exp[j], linestyle=lst_exp[j])
            for j in range (len(index))]
        ax.text(*coord_text, subfigures[i], transform = ax.transAxes)
        legend_box = [0.48, 0.65]
        if exps[i] == 'V03':
            legend_box = [1., 0.5]
        else:
            legend_box = [1.0, 0.3]
        plt.legend(loc='center right', bbox_to_anchor=legend_box, handles=legend_elements, ncol=1, fontsize=13, handlelength=2.4)
        plt.savefig(path + f"Figures-pool_model_real_data_exp_{names[i]}.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='-.', color=colors_all['T'])
        ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacteriocin [AU/mL]')
        ax.set_xlim(-0.05, np.max(t_model))
        legend_elements = [Line2D([0], [0], color=colors_all['T'], label='Bacteriocin', marker='X', linestyle='-.')]
        legend_box = [0.48, 0.75]
        plt.legend(handles=legend_elements, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)
        ax.text(*coord_text, r'\textbf{F}', transform = ax.transAxes)
        plt.savefig(path + f"Figures-pool_model_real_data_BAC_{names[i]}.png", bbox_inches="tight")
        plt.close(fig)

def get_param_dfs(path, path2):
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    names = ['Ls23K', 'Lm', 'Ls23K-Lm']
    df_names = [f'dataframe_poolpaper_{name}.pkl' for name in names]
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    return param_opt, dfs, df_optim2


if __name__ == "__main__":
    n_cl = 4
    relnoise = 0.

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/'
    #path = 'pool_paper_casestudy/out/all_togetherexps_withpH_myderiv/'
    #path2 = 'pool_paper_casestudy/out/all_togetherexps_withpH_myderiv/'
    add_name = ''
    model = ode_model_coculture3

    param_opt, dfs, df_optim2 = get_param_dfs(path, path2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)

    names = ['Ls23K', 'Lm', 'Ls23K-Lm']
    n_exps = len(names)
    temps = [2.0 for _ in range(len(names))]
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))

    #plot_paper_figures(param_opt, dfs, path=path2, add_name=add_name)
    plot_cases_separately(param_opt, dfs, model, path=path2, add_name=add_name)
    exit()

    # With pH term
    path = 'pool_paper_casestudy/out/all_x_LA_BAC_with_death_wo_pH_latest/'
    path2 = 'pool_paper_casestudy/out/all_x_LA_BAC_with_death_wo_pH_latest/'
    param_opt1, dfs1, _ = get_param_dfs(path, path2)

    # Without pH term
    path = 'pool_paper_casestudy/out/all_wo_pH_influence/'
    path2 = 'pool_paper_casestudy/out/all_wo_pH_influence/'
    param_opt2, dfs2, _ = get_param_dfs(path, path2)

    plot_comparision(param_opt1, dfs1,  param_opt2, dfs2, path='pool_paper_casestudy/out/', add_name='_pH_influence')
    exit()


    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)

    for i in range (len(data)):
        #if exps[i] != 'LsCTC494' and exps[i] != 'LsCTC494-Lm' and exps[i] != 'V01' and exps[i] != 'V05':
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            #param_ode_new[2*4 + 2 + 3+1] = 0.
            param_ode_new[4*3+3+3] = 0.
            plot_all_curves(param_ode_new, x0_vals[n_cl*i:n_cl*(i+1)], data=data[i], path=path2, add_name=f'_estim_realdata_{names[i]}')
        else:
            plot_all_curves(param_ode, x0_vals[n_cl*i:n_cl*(i+1)], data=data[i], path=path2, add_name=f'_estim_realdata_{names[i]}')

    
    calibr_setup = {
        "model": model,
        "output_path": path2,
        "n_cl": n_cl,
        "dfs": [dfs],
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps,
        "exp_temps": {exp: temp for exp, temp in zip(exps, temps)},
    }
    data_array = extract_observables_from_df(calibr_setup["dfs"])
    calibr_setup["data_array"] = data_array
    cost(param_opt, calibr_setup, None)
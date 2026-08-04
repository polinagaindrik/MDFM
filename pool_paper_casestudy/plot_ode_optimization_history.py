import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_comparision(param_opt1, dfs1,  param_opt2, dfs2, path='', add_name=''):
    names = ['Ls23K', 'LsCTC494', 'LmCTC1034', 'Ls23K-LmCTC1034', 'LsCTC494-LmCTC1034']
    n_exps = len(names)
    coord_text = (0.04, 0.88)
    model = ode_model_coculture2
    
    exp_indexes = [3, 4 ,0, 1, 2]
    lbls = ['Ls-23K', 'Ls-CTC494', 'Lm-CTC1034', 'Lactic Acid', 'pH']
    mrkrs = ['o', 'o',  'o', '^', 'x']
    lst = ['solid', 'solid', 'solid', 'dashed', '']
    clrs = [colors_all['N_LsCTC494'], colors_all['N_LsCTC494co'], colors_all['N_Lm_withT'], colors_all['T_A'], colors_all['N_Ls23K']]
    clr_indexes = [[0, 3, 4], [1, 3, 4], [2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
    obs_count_indexes = [[0], [0], [1], [0, 1], [0, 1]]
    subfigures = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}']

    for i in exp_indexes:
        index = clr_indexes[i]
        clrs_exp = [clrs[ind] for ind in index]
        lbls_exp = [lbls[ind] for ind in index]
        mrkrs_exp = [mrkrs[ind] for ind in index]
        lst_exp = [lst[ind] for ind in index]
        obs_count_ind_exp = obs_count_indexes[i]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        for l, (param_opt, dfs) in enumerate(zip([param_opt1, param_opt2], [dfs1, dfs2])):
            alpha = 1. - l*0.3
            if l == 0:
                linestyle = 'solid'
            else:
                linestyle = 'dotted'
            
            exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
            x0_vals = param_opt[:n_cl*n_exps]
            param_ode = list(param_opt[n_cl*n_exps:])
            param_ode_new = np.copy(param_ode)
            #param_ode_new[2*3 + 2 + 3+1] = 0.
            param_ode_new[4*3+3+3] = 0.
            days, [obs_x] = extract_observables_from_df([dfs])
            t_model = np.linspace(days[0], days[-1]+5, 100)

            x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
            pH_series = np.array([obs_x[i][-1], days]).T
            if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
                x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
            else:
                x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
            obs_model = observable(t_model, x_sol)

            for k in range(len(index)-2):
                ax.plot(t_model, obs_model[obs_count_ind_exp[k]], label='Ls-CTC494', color=clrs_exp[k], linewidth=2+l*2, linestyle=linestyle, alpha=alpha)
                ax.scatter(days, obs_x[i][obs_count_ind_exp[k]], marker=mrkrs_exp[k], color=clrs_exp[k])
            
            ax2.plot(t_model, obs_model[3], linewidth=2+l*2, color=clrs_exp[k+1], linestyle=linestyle, alpha=alpha)
            ax2.scatter(days, obs_x[i][3], color=clrs_exp[k+1], marker=mrkrs_exp[k+1])

        ax.set_xlim(-0.1, np.max(t_model))
        ax.set_yscale('log')
        ax.set_title(names[i])
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
        fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'pH; Lactic Acid [g/L]')

        legend_elements = [
            Line2D([0], [0], color=clrs_exp[j], label=lbls_exp[j]+' (with pH)', marker=mrkrs_exp[j], linestyle='solid', linewidth=2)
            for j in range (len(index)-1)] + [Line2D([0], [0], color=clrs_exp[j], label=lbls_exp[j]+' (without pH)', alpha=0.7, linestyle='dotted', linewidth=4)
            for j in range (len(index)-1)]
        ax.text(*coord_text, subfigures[i], transform = ax.transAxes)
        legend_box = [0.48, 0.65]
        plt.legend(handles=legend_elements, ncol=1, fontsize=13, handlelength=2.4, loc='lower right', framealpha=0.5)
        plt.savefig(path + f"Figures-pool_model_real_data_comparison_exp_{names[i]}.png", bbox_inches="tight")
        plt.close(fig)   


def plot_cases_separately(param_opt, dfs, model, path='', add_name=''):
    names = ['Ls23K', 'LsCTC494', 'LmCTC1034', 'Ls23K-LmCTC1034', 'LsCTC494-LmCTC1034']
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

    exp_indexes = [3, 4 ,0, 1, 2]
    lbls = ['Ls-23K','Ls-CTC494',  'Lm-CTC1034', 'Lactic Acid', 'pH']
    mrkrs = ['o', 'o',  'o', '^', 'x']
    lst = ['solid', 'solid', 'solid', 'dashed', '']
    clrs = [colors_all['N_LsCTC494'], colors_all['N_LsCTC494co'], colors_all['N_Lm_withT'], colors_all['T_A'], colors_all['N_Ls23K']]
    clr_indexes = [[0, 3, 4], [1, 3, 4], [2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
    obs_count_indexes = [[0], [0], [1], [0, 1], [0, 1]]
    subfigures = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}']

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
            ax.plot(t_model, obs_model[obs_count_ind_exp[k]], label='Ls-CTC494', color=clrs_exp[k], linewidth=3, linestyle=lst_exp[k])
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


    
def plot_paper_figures(param_opt, dfs, model, path='', add_name=''):
    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    n_exps = len(names)
    coord_text = (0.04, 0.88)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    param_ode_new[2*4 + 2 + 3+1] = 0.
    #param_ode_new[4*3+3+3] = 0. # another model
    days, [obs_x] = extract_observables_from_df([dfs])
    t_model = np.linspace(days[0], days[-1]+5, 100)

    # Figure A w/o BAC: Ls23K + Lm
    exp_indexes = [1, 2, 3,]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)
        if i == 1:
            ax.plot(t_model, obs_model[0], label='Monoculture: Ls23K', linestyle='dashed', color=colors_all['N_lambd_1e-3_omega_0'], linewidth=3.5)
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_lambd_1e-3_omega_0'])
        elif i == 2:
            ax.plot(t_model, obs_model[1], label='Monoculture: Lm', color=colors_all['N_tempshift_10'], linestyle='dashed', linewidth=3.5)
            ax.scatter(days, obs_x[i][1], marker='^', color=colors_all['N_tempshift_10'])
        else:
            ax.plot(t_model, obs_model[0], label='Coculture: Ls23K', color=colors_all['N_B'])
            ax.plot(t_model, obs_model[1], label='Coculture: Lm', linestyle='solid', color=colors_all['N_A'])
            ax.scatter(days, obs_x[i][0], marker='o', color=colors_all['N_B'])
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_A'])
    ax.set_yscale("log")
    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')

    legend_elements = [Line2D([0], [0], color=colors_all['N_lambd_1e-3_omega_0'], label='Ls-23K (mono)', marker='^', linestyle='dashed'),
    Line2D([0], [0], color=colors_all['N_tempshift_10'], label='Lm-CTC1034 (mono)', marker='^', linestyle='dashed'),
    Line2D([0], [0], color=colors_all['N_B'], label='Ls-23K (co)', marker='o'),
    Line2D([0], [0], color=colors_all['N_A'], label='Lm-CTC1034 (co)', marker='o')]
    plt.legend(handles=legend_elements)
    ax.text(*coord_text, r'\textbf{A}', transform = ax.transAxes)
    
    plt.savefig(path + f"Figures-pool_model-real_data-Ls23K-Lm.png", bbox_inches="tight")
    plt.close(fig)

    # Figure B: LsCTC494 + Lm
    exp_indexes = [0, 2, 4]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)
        if i == 0:
            ax.plot(t_model, obs_model[0], label='Ls-CTC494 (mono)', linestyle='dashed', color=colors_all['N_lambd_1e-3_omega_0'], linewidth=3.5)
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_lambd_1e-3_omega_0'])
        elif i == 2:
            ax.plot(t_model, obs_model[1], label='Lm-CTC1034 (mono)', color=colors_all['N_tempshift_10'], linestyle='dashed', linewidth=3.5)
            ax.scatter(days, obs_x[i][1], marker='^', color=colors_all['N_tempshift_10'])
        else:
            ax.plot(t_model, obs_model[0], label='Ls-CTC494 (co)', color=colors_all['N_B'])
            ax.plot(t_model, obs_model[1], label='Lm-CTC1034 (co)', linestyle='solid', color=colors_all['N_A'])
            #ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dashed', color=colors_all['T'])

            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_B'])
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_A'])
            #ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'Bacteriocin [AU/mL]')
    ax2.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='-.', color=colors_all['T'])
    ax2.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax.set_yscale("log")
    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
    legend_elements = [
        Line2D([0], [0], color=colors_all['N_lambd_1e-3_omega_0'], label='Ls-CTC494 (mono)', marker='^', linestyle='dashed'),
        Line2D([0], [0], color=colors_all['N_tempshift_10'], label='Lm-CTC1034 (mono)', marker='^', linestyle='dashed'),
        Line2D([0], [0], color=colors_all['N_B'], label='Ls-CTC494 (co)', marker='o'),
        Line2D([0], [0], color=colors_all['N_A'], label='Lm-CTC1034 (co)', marker='o'),
        Line2D([0], [0], color=colors_all['T'], label='Bacteriocin', marker='X', linestyle='-.')
        ]
    legend_box = [0.48, 0.75]
    plt.legend(handles=legend_elements, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)
    ax.text(*coord_text, r'\textbf{B}', transform = ax.transAxes)
    
    plt.savefig(path + f"Figures-pool_model-real_data-LsCTC494-Lm.png", bbox_inches="tight")
    plt.close(fig)

    # All together:
    exp_indexes = [0, 1, 2, 3, 4, ]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)
        if i == 0:
            ax.plot(t_model, obs_model[0], label='Ls-CTC494', linestyle='dashed', color=colors_all['N_LsCTC494'], linewidth=3)
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_LsCTC494'])
        elif i == 1:
            ax.plot(t_model, obs_model[0], label='Ls-23K', linestyle='dashed', color=colors_all['N_Ls23K'], linewidth=3)
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_Ls23K'])
        elif i == 2:
            ax.plot(t_model, obs_model[1], label='Lm-CTC1034', color=colors_all['N_Lm'], linestyle='dashed', linewidth=3)
            ax.scatter(days, obs_x[i][1], marker='^', color=colors_all['N_Lm'])
        elif i == 3:
            ax.plot(t_model, obs_model[0], label='Ls-23K (co)', color=colors_all['N_Ls23Kco'])
            ax.plot(t_model, obs_model[1], label='Lm (co Ls-23K)', linestyle='solid', color=colors_all['N_Lm_woT'])
            #ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dashed', color=colors_all['T'])

            ax.scatter(days, obs_x[i][0], marker='D', color=colors_all['N_Ls23Kco'])
            ax.scatter(days, obs_x[i][1], marker='D', color=colors_all['N_Lm_woT'])
            #ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])

        else:
            ax.plot(t_model, obs_model[0], label='Ls-CTC494 (co)', color=colors_all['N_LsCTC494co'])
            ax.plot(t_model, obs_model[1], label='Lm (co Ls-CTC494)', linestyle='solid', color=colors_all['N_Lm_withT'])
            #ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dashed', color=colors_all['T'])

            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_LsCTC494co'])
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_Lm_withT'])
            #ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'Bacteriocin [AU/mL]')
    ax2.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dotted', color=colors_all['T'])
    ax2.set_ylim(-150, 5100)
    ax2.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax.set_yscale("log")
    ax.set_xlim(-0.1, 57)
    ax.set_ylim(2*10**(-2), 2*10**9)
    ax.text(*coord_text, r'\textbf{A}', transform = ax.transAxes)
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
    
    legend_elements = [
        Line2D([0], [0], color=colors_all['N_Ls23K'], label='Ls-23K', marker='^', linestyle='dashed'),
        Line2D([0], [0], color=colors_all['N_Ls23Kco'], label='Ls-23K (co. Lm)', marker='D'),

        Line2D([0], [0], color=colors_all['N_LsCTC494'], label='Ls-CTC494', marker='^', linestyle='dashed'),
        Line2D([0], [0], color=colors_all['N_LsCTC494co'], label='Ls-CTC494 (co. Lm)', marker='o'),

        Line2D([0], [0], color=colors_all['N_Lm'], label='Lm', marker='^', linestyle='dashed'),

        Line2D([0], [0], color=colors_all['N_Lm_woT'], label='Lm (co. 23K)', marker='D'),
        Line2D([0], [0], color=colors_all['N_Lm_withT'], label='Lm (co. CTC494)', marker='o'),
        #Line2D([0], [0], color=colors_all['T'], label='Bacteriocin', marker='x', linestyle='-.')
    ]
    legend_box = [0.57, 0.62]
    ax.legend(handlelength=2.4, handles=legend_elements, ncol=1, fontsize=11, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)

    legend_elements = [
        Line2D([0], [0], color=colors_all['T'], label='Bacteriocin', marker='X', linestyle='dotted')
    ]
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=12, handlelength=2.4)
    plt.savefig(path + f"Figures-pool_model_real_data_all_count.pdf", bbox_inches="tight")
    plt.close(fig)
  
    # Plotting lactic acid:
    # All
    exp_indexes = [3, 4 ,0, 1, 2]
    lbls = ['Ls-23K', 'Ls-CTC494', 'Lm-CTC1034', 'Lm/Ls-23K (co.)', 'Lm/Ls-CTC494 (co.)']
    mrkrs = ['^', '^', '^', 'D', 'o']
    lst = ['dashed','dashed', 'dashed', 'solid', 'solid']
    clrs = [colors_all['N_Ls23K'], colors_all['N_LsCTC494'], colors_all['N_Lm'], colors_all['N'], colors_all['N_lambd_1e-2_omega_0']]
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)

        ax.plot(t_model, obs_model[3], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax.scatter(days, obs_x[i][3], color=clrs[i], marker=mrkrs[i])

        ax.scatter(days, obs_x[i][4], color=clrs[i], marker='x')

    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Lactic Acid [g/L]')
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', 'pH')
    ax.set_ylim(-.5, 6.)
    ax2.set_ylim(-.5, 6.)
    legend_elements = [
        Line2D([0], [0], color=clrs[i], label=lbls[i], marker=mrkrs[i], linestyle=lst[i])
        for i in exp_indexes]
    ax.text(*coord_text, r'\textbf{B}', transform = ax.transAxes)
    legend_box = [0.48, 0.65]
    plt.legend(handles=legend_elements, ncol=1, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure, fontsize=13, handlelength=2.4)
    
    plt.savefig(path + f"Figures-pool_model_real_data_LA_pH.pdf", bbox_inches="tight")
    plt.close(fig)


def cost_res(param, calibr_setup):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    param_ode = param[n_cl*n_exps:]
    param_ode_new = np.copy(param_ode)
    x0_vals = param[:n_cl*n_exps]

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
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode_new, x_max[i])
        else:
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return calibr_setup["aggregation_func"]([ll_x])


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, x_max):
    # TODO mb: do we need to fit also data for BAC, LA (pH)
    # Then obs_x -> obs_x+m
    # + pH instead of temp?
    model = calibr_setup["model"]
    days, [obs_x] = calibr_setup["data_array"]
    pH_series = np.array([obs_x[i][-1], days]).T
    const = [pH_series, n_cl]

    C0 = set_initial_vals(x0, None, n_cl, pH0=obs_x[i][-1][0])
    #np.concatenate((np.array(x0), np.array([0., 0.]), np.array([1., 0., 0., 6.])))
    C = fm.mdl.model_ODE_solution(model, days, param_ode, C0, const, t0=days[0])
    obs_model = observable(days, C)
    ll_x0 = [
        (obs_x[i][0] - obs_model[0]) ** 2 / x_max[0], #  G
        (obs_x[i][1] - obs_model[1]) ** 2 / x_max[1],
        (obs_x[i][2] - obs_model[2]) ** 2 / np.max(x_max[2]), # BAC
        (obs_x[i][3] - obs_model[3]) ** 2, #/ x_max[3], # LA
        #(obs_x[i][4] - obs_model[4]) ** 2, #/ x_max[3], # pH
    ]
    return np.array(ll_x0)



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

    names = ['Ls23K', 'LsCTC494', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
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
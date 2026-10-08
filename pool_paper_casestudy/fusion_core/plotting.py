"""
``plot_cost_function`` is ported from
``fusion_model/plotting/plotting_parameters.py``.

Everything else here (the pool-paper color palette / figure-size / rcParams
styling, and its plotting functions) was moved from
``pool_paper_casestudy/pool_model_functions.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import dtf
from . import mdl


def plot_cost_function(df_optim, path='', add_name=''):
    """Plot the optimization-history cost curve (`iteration` vs `cost`, log-y)."""
    fig, ax = plt.subplots()
    ax.plot(df_optim['iteration'], df_optim['cost'])
    ax.set_xlabel('optimization step', fontsize=12)
    ax.set_ylabel('cost function', fontsize=12)
    ax.set_yscale('log')
    plt.savefig(path + f'cost_plot{add_name}.png')  # , bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Pool-paper style constants
# ----------------------------------------------------------------------
colors_all = {
        'R': '#808080',
        'N_A':'#D06062',
        'N_B': '#4E89B1',
        'N':'#7E57A5',
        'T':'#99582A',
        'T_A':'#c79758',
        'N_lambd_1e-2_omega_0':'#E2B100',
        'N_lambd_1e-3_omega_0':'#386641',
        'N_lambd_1e-3_omega_0_5':'#0982A4',
        'N_wo_tempshift':'#679E48',
        'N_tempshift_10':'#ED733E',
        'N_tempshift_10_5_15':'#C3568A',
        'N_Lm': '#ED733E',
        'N_Lm_woT':'#D70040',
        'N_Ls23K': '#679E48',
        'N_Ls23Kco': '#386641',
        'N_Lm_withT':'#D06062',
        'N_LsCTC494': '#4E89B1',
        'N_LsCTC494co': '#00356B',
    }

figsize_default = (6.5, 4.0)
figsize_default_small = (6.5, 2.0)
figsize_default2subpl = (13, 4.0)

plt.rcParams['figure.dpi'] = 400
plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

plt.rcParams['legend.fontsize'] = 15.
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.handlelength'] = 1.8
plt.rcParams['axes.prop_cycle'] = plt.cycler(linewidth=[2.5])
plt.rcParams['font.size'] = 15
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)


def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    return fig, ax


########### pH(LA) # function
def pH_LA_dependence(days, LA_data, pH_data, add_name='', path=''):
    pH0 = pH_data[0]
    K_a = 1.38*10**(-4)
    pH = pH0 + np.log(- K_a + np.sqrt(K_a**2 + 4 * K_a*LA_data)/2)
    #print(- K_a + np.sqrt(K_a**2 + 4 * K_a*LA_data))
    fig, ax = plt.subplots()
    ax.scatter(days, pH, label='pH(LA)')
    ax.scatter(days, pH_data, label='pH_data', marker='x')
    ax.scatter(days, LA_data, label='LA_data', marker='x')
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)
    return pH


############## Plotting ######################
def plot_all_curves(param_ode, x10, model=None, obs=None, data=None, path='', add_name=''):
    if model is None:
        model = mdl.ode_model_coculture
    if obs is None:
        obs = mdl.observable
    clrs = [colors_all['N_A'], colors_all['N_B'], colors_all['R'], colors_all['T'], colors_all['N']]
    n_cl = 3
    if data is not None:
        days, [obs_x] = dtf.extract_observables_from_df([data])
    t = np.linspace(days[0], days[-1], 100)
    x0 = mdl.set_initial_vals(x10, None, n_cl, pH0=obs_x[0][-1][0])
    pH_series = np.array([obs_x[0][-1], days]).T
    x_sol = mdl.model_ODE_solution(model, t, param_ode, x0, [pH_series, n_cl])
    obs_model = obs(days, x_sol)
    lbls = ["ls", "lm", "BAC", "LA", "pH"]
    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(t, obs_model[i], label=lbls[i], color=clrs[i])
        if data is not None:
            ax.scatter(days, obs_x[0][i], label=lbls[i]+'_data', marker='x', color=clrs[i])
    # ax.plot(t, x_sol[2], label='R')

    ax.plot(t, x_sol[4], label='R', linestyle='dashed', color=clrs[4])
    ax.plot(t, x_sol[2], label='lm_sen', linestyle='dotted', color=clrs[2])
    ax.plot(t, x_sol[3], label='lm_res', linestyle='dotted', color=clrs[3])
    ax.set_yscale("log")
    #ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path + f"x_sol_R{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[2], label=lbls[2], color=clrs[2])
    if data is not None:
        ax.scatter(days, obs_x[0][2], label=lbls[2]+'_data', marker='x', color=clrs[2])
    plt.legend()
    plt.savefig(path + f"BAC{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[3], label=lbls[3], color=clrs[3])
    if data is not None:
        ax.scatter(days, obs_x[0][3], label=lbls[3]+'_data', marker='x', color=clrs[3])
        ax.scatter(days, obs_x[0][4], label=lbls[4]+'_data', marker='x', color=clrs[4])
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)


######
def plot_cases_separately(param_opt, dfs, model, path='', add_name='', exp_indexes=[0, 1, 2, 3, 4]):
    n_cl = 4
    names = ['Ls23K', 'LsCTC494', 'LmCTC1034', 'Ls23K-LmCTC1034', 'LsCTC494-LmCTC1034']
    n_exps = len(names)
    coord_text = (0.04, 0.88)
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)

    if model == mdl.ode_model_coculture:
        param_ode_new[2*4 + 2 + 3+1] = 0.
    elif model == mdl.ode_model_coculture2:
        param_ode_new[4*3+3+3] = 0.
    elif model == mdl.ode_model_coculture3:
        param_ode_new[4*3+3+3] = 0.
    elif model == mdl.ode_model_coculture_wopH:
        param_ode_new[3+3+3] = 0.
    elif model == mdl.ode_model_coculture_wopH_MM:
        param_ode_new[8] = 0.
    elif model == mdl.ode_model_coculture_wopH_expsat:
        param_ode_new[8] = 0.
    elif model == mdl.ode_model_coculture_withpH_MM:
        param_ode_new[17] = 0.

    days, [obs_x] = dtf.extract_observables_from_df([dfs])
    t_model = np.linspace(days[0], days[-1]+5, 100)
    obs_model = np.zeros((len(exps), np.shape(obs_x)[0], len(t_model)))
    obs_model_rmse = np.zeros(np.shape(obs_x))
    # exp_indexes = [3, 4 ,0, 1, 2]
    lbls = ['Ls-23K','Ls-CTC494',  'Lm-CTC1034', 'Lactic Acid']#, 'pH']
    mrkrs = ['o', 'o',  'o', '^', 'x']
    lst = ['solid', 'solid', 'solid', 'dashed', '']
    clrs = [colors_all['N_LsCTC494'], colors_all['N_LsCTC494co'], colors_all['N_Lm_withT'], colors_all['T_A'], colors_all['N_Ls23K']]
    #clr_indexes = [[0, 3, 4], [1, 3, 4], [2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
    clr_indexes = [[0, 3], [1, 3], [2, 3], [0, 2, 3], [1, 2, 3]]
    obs_count_indexes = [[0], [0], [1], [0, 1], [0, 1]]
    subfigures = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}']
    for j, i in enumerate(exp_indexes):
        index = clr_indexes[i]
        clrs_exp = [clrs[ind] for ind in index]
        lbls_exp = [lbls[ind] for ind in index]
        mrkrs_exp = [mrkrs[ind] for ind in index]
        lst_exp = [lst[ind] for ind in index]
        obs_count_ind_exp = obs_count_indexes[i]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        x0 = mdl.set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = mdl.model_ODE_solution(model, t_model, param_ode_new, x0, [pH_series, n_cl])
            x_sol_rmse  = mdl.model_ODE_solution(model, days, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = mdl.model_ODE_solution(model, t_model, param_ode, x0, [pH_series, n_cl])
            x_sol_rmse  = mdl.model_ODE_solution(model, days, param_ode, x0, [pH_series, n_cl])
        obs_model[i] = mdl.observable(t_model, x_sol)
        obs_model_rmse[i] = mdl.observable(days, x_sol_rmse)

        for k in range(len(index)-1):
            ax.plot(t_model, obs_model[i][obs_count_ind_exp[k]], label='Ls-CTC494', color=clrs_exp[k], linewidth=3, linestyle=lst_exp[k])
            ax.scatter(days, obs_x[i][obs_count_ind_exp[k]], marker=mrkrs_exp[k], color=clrs_exp[k])

        ax2.plot(t_model, obs_model[i][3], linewidth=3, color=clrs_exp[k+1], linestyle=lst_exp[k+1])
        ax2.scatter(days, obs_x[i][3], color=clrs_exp[k+1], marker=mrkrs_exp[k+1])
        #ax2.scatter(days, obs_x[i][4], color=clrs_exp[k+2], marker=mrkrs_exp[k+2])

        ax.set_xlim(-0.05, np.max(t_model))
        ax.set_yscale('log')
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
        #fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'pH; Lactic Acid [g/L]')
        fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'Lactic Acid [g/L]')
        ax2.set_ylim(-0.5, 6.5)
        legend_elements = [
            Line2D([0], [0], color=clrs_exp[j], label=lbls_exp[j], marker=mrkrs_exp[j], linestyle=lst_exp[j])
            for j in range (len(index))]
        ax.text(*coord_text, subfigures[j], transform = ax.transAxes)
        legend_box = [0.48, 0.65]
        if exps[i] == 'V03':
            legend_box = [1., 0.45]
        elif exps[i] == 'V05':
            legend_box = [1.0, 0.35]
        else:
            legend_box = [1.0, 0.2]
        plt.legend(loc='center right', bbox_to_anchor=legend_box, handles=legend_elements, ncol=1, fontsize=15, handlelength=2.8)
        plt.savefig(path + f"Figures-pool_model_real_data_exp_{names[i]}"+add_name+".pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t_model, obs_model[i][2], label='Bacteriocin', linestyle='-.', color=colors_all['T'])
        ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacteriocin [AU/mL]')
        ax.set_xlim(-0.05, np.max(t_model)-3)
        #legend_elements = [Line2D([0], [0], color=colors_all['T'], label='Bacteriocin', marker='X', linestyle='-.')]
        legend_elements = [Line2D([0], [0], color=colors_all['T'], label='Model', marker='', linestyle='-.'),
        Line2D([0], [0], color=colors_all['T'], label='Experimental data', marker='X', linestyle='')
        ]
        legend_box = [0.5, 0.4]
        plt.legend(handles=legend_elements, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure, fontsize=15, handlelength=2.8)
        ax.text(*coord_text, r'\textbf{F}', transform = ax.transAxes)
        plt.savefig(path + f"Figures-pool_model_real_data_BAC_{names[i]}"+add_name+".pdf", bbox_inches="tight")
        plt.close(fig)
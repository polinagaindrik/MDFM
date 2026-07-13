import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def set_initial_vals(x10, temps, n_cl, pH0=6.):
    return np.concatenate((x10, [1., 0., 0., pH0]))

def ode_model_coculture(t, x, param, x0, ode_args):
    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    (x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls_opt, mu_lm_opt,
    pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt,
    omega_ls_exp, omega_lm_exp,
    omegaT_lm_exp, k_T_inhib0, n,
    N_texp,
    kappa_T_0, 
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp,
    q_acid) = param

    mu_ls = mu_ls_opt * (pH - pH_ls_min) / (pH_ls_opt - pH_ls_min)
    mu_lm = mu_lm_opt * (pH - pH_lm_min) / (pH_lm_opt - pH_lm_min)
    #mu_ls = mu_ls_opt**2 * (pH - pH_ls_min)**2
    #mu_lm = mu_lm_opt**2 * (pH - pH_lm_min)**2

    N_t = 10**N_texp
    omega_ls = 10**(-3) * omega_ls_exp
    omega_lm = 10**(-3) * omega_lm_exp
    omegaT_lm = omegaT_lm_exp
    kappa_T = 10**(-5) * kappa_T_0
    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm = 10**np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp])

    k_T_inhib = k_T_inhib0
    toxin_death = omegaT_lm * x_lm_sen * T**n / (k_T_inhib**n + T**n) #omegaT_lm * x_lm_sen * T #

    return [
        (mu_ls * R - omega_ls) * x_ls23K,
        (mu_ls * R - omega_ls) * x_lsCTC494,
        (mu_lm * R  - omega_lm) * x_lm_sen - toxin_death,
        (mu_lm * R  - omega_lm) * x_lm_res,
        -(mu_ls / N_t)*R*x_ls23K - (mu_ls / N_t)*R*x_lsCTC494 - (mu_lm / N_t)*R*x_lm_sen - (mu_lm / N_t)*R*x_lm_res,
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K*x_ls23K + kappa_LA_ls23K_2*R*x_ls23K) + (kappa_LA_lsCTC494*x_lsCTC494 + kappa_LA_lsCTC494_2*R*x_lsCTC494) +
        + kappa_LA_lm  * (x_lm_sen+x_lm_res),  # *R but wo R the curves look better
        0.#- q_acid * LA
    ]
    
def observable(t, x):
    n = np.array([x[0]+x[1], x[2]+x[3]])
    obs = np.concatenate((n, x[5:])) # mb add pH
    return obs


def pH_func(t, pH_series):
    # pH_series = [[pH1, t1], [pH2, t2], [pH3, t3], ...] (n_times x 2)
    pH_arr, time_arr = np.array(pH_series).T
    diff = time_arr - t
    return pH_arr[np.argmin(np.abs(diff))]


def extract_observables_from_df(dfs):
    (df_x,) = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split("_")[-2]) for f in df_x.columns]))
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days_x)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days_x):
            df0 = df_x.filter(like=exp).filter(like=f"_{int(d):02d}_")
            if np.shape(df0)[-1] != 0.0:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan * np.ones((np.shape(df_x)[0]))
    return days_x, [obs_x]

def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    return fig, ax


def plot_all_curves(param_ode, x10, data=None, path='', add_name=''):
    [days, obs_x, pH_series], [t, obs_model, x_sol] = get_data_for_plotting(param_ode, x10, data=data)
    lbls = ["ls", "lm", "BAC", "LA", "pH"]
    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(t, obs_model[i], label=lbls[i])
        if data is not None:
            ax.scatter(days, obs_x[0][i], label=lbls[i]+'_data', marker='x')
    # ax.plot(t, x_sol[2], label='R')

    #ax.plot(t, x_sol[4], label='R', linestyle='dotted')
    ax.plot(t, x_sol[2], label='lm_sen', linestyle='dotted')
    ax.plot(t, x_sol[3], label='lm_res', linestyle='dotted')
    ax.set_yscale("log")
    #ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path + f"x_sol_R{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[2], label=lbls[2])
    if data is not None:
        ax.scatter(days, obs_x[0][2], label=lbls[2]+'_data', marker='x')
    plt.legend()
    plt.savefig(path + f"BAC{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[3], label=lbls[3])
    ax.plot(t, obs_model[4], label=lbls[4])
    if data is not None:
        ax.scatter(days, obs_x[0][3], label=lbls[3]+'_data', marker='x')
        ax.scatter(days, obs_x[0][4], label=lbls[4]+'_data', marker='x')
        plt.legend()
        plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)


def get_data_for_plotting(param_ode, x10, data=None):
    n_cl = 4
    if data is not None:
        days, [obs_x] = extract_observables_from_df([data])
    t = np.linspace(days[0], days[-1], 100)
    x0 = set_initial_vals(np.array(x10), None, n_cl, pH0=obs_x[0][-1][0])
    pH_series = np.array([obs_x[0][-1], days]).T
    x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t, param_ode, x0, [pH_series, n_cl])
    obs_model = observable(days, x_sol)
    return [days, obs_x, pH_series], [t, obs_model, x_sol]


def plot_paper_figures(param_opt, dfs, path='', add_name=''):
    names = ['LsCTC494', 'Ls23K', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    n_exps = len(names)
    coord_text = (0.04, 0.88)

    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))

    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    param_ode_new[2*3 + 2 + 3+1] = 0.

    days, [obs_x] = extract_observables_from_df([dfs])
    t_model = np.linspace(days[0], days[-1]+1.5, 100)

    # Figure A w/o BAC: Ls23K + Lm
    exp_indexes = [1, 2, 3,]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
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
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
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

    '''
    # All together:
    exp_indexes = [0, 1, 2, 3, 4]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)
        if i == 0:
            ax.plot(t_model, obs_model[0], label='Monoculture: LsCTC494', linestyle='dashed', color=colors_all['N_A'])
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_A'])
        elif i == 1:
            ax.plot(t_model, obs_model[0], label='Monoculture: Ls23K', linestyle='dashed', color=colors_all['N_A'])
            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_A'])
        elif i == 2:
            ax.plot(t_model, obs_model[1], label='Monoculture: Lm', color=colors_all['N_B'], linestyle='dashed')
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_B'])
        elif i == 3:
            ax.plot(t_model, obs_model[0], label='Coculture: Ls23K', color=colors_all['N_lambd_1e-2_omega_0'])
            ax.plot(t_model, obs_model[1], label='Coculture: Lm', linestyle='solid', color=colors_all['N_lambd_1e-3_omega_0'])
            #ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dashed', color=colors_all['T'])

            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_lambd_1e-2_omega_0'])
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_lambd_1e-3_omega_0'])
            #ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])

        else:
            ax.plot(t_model, obs_model[0], label='Coculture: LsCTC494', color=colors_all['N_lambd_1e-2_omega_0'])
            ax.plot(t_model, obs_model[1], label='Coculture: Lm', linestyle='solid', color=colors_all['N_lambd_1e-3_omega_0'])
            #ax.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dashed', color=colors_all['T'])

            ax.scatter(days, obs_x[i][0], marker='^', color=colors_all['N_lambd_1e-2_omega_0'])
            ax.scatter(days, obs_x[i][1], marker='o', color=colors_all['N_lambd_1e-3_omega_0'])
            #ax.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', r'Bacteriocin [AU/mL]')
    ax2.plot(t_model, obs_model[2], label='Bacteriocin', linestyle='dotted', color=colors_all['T'])
    ax2.scatter(days, obs_x[i][2], marker='X', color=colors_all['T'])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax.set_yscale("log")
    ax.set_xlim(-0.1, 55)
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count [CFU/mL]')
    ax.legend()
    ax2.legend()
    plt.savefig(path + f"Figures-pool_model-real_data-all_count.png", bbox_inches="tight")
    plt.close(fig)
    '''
    # Plotting lactic acid:
    # All
    exp_indexes = [0, 1, 2, 3, 4]
    lbls = ['Ls-CTC494 (mono)', 'Ls-23K (mono)', 'Lm-CTC1034 (mono)', 'Ls-23K + Lm-CTC1034 (co)', 'Ls-CTC494 + Lm-CTC1034 (co)']
    mrkrs = ['^', '^', '^', 'o', 'o']
    lst = ['dashed','dashed', 'dashed', 'solid', 'solid']
    clrs = [colors_all['R'], colors_all['N'], colors_all['N_tempshift_10_5_15'], colors_all['N_wo_tempshift'], colors_all['N_lambd_1e-3_omega_0_5']]
    fig, ax = plt.subplots()
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)

        ax.plot(t_model, obs_model[3], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax.scatter(days, obs_x[i][3], color=clrs[i], marker=mrkrs[i])

    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Lactic Acid [g/L]')
    ax.set_ylim(-1, 7.)
    legend_elements = [
        Line2D([0], [0], color=clrs[0], label=lbls[0], marker='^', linestyle='dashed'),
        Line2D([0], [0], color=clrs[1], label=lbls[1], marker='^', linestyle='dashed'),
        Line2D([0], [0], color=clrs[2], label=lbls[2], marker='o'),
        Line2D([0], [0], color=clrs[3], label=lbls[3], marker='o'),
        Line2D([0], [0], color=clrs[4], label=lbls[4], marker='o')
        ]
    plt.legend(handles=legend_elements)
    ax.text(*coord_text, r'\textbf{C}', transform = ax.transAxes)
    
    plt.savefig(path + f"Figures-pool_model-real_data-LA.png", bbox_inches="tight")
    plt.close(fig)

    # Plotting lactic acid:
    # For mono cultures
    exp_indexes = [1, 2, 3]
    lbls = ['Ls-CTC494 (mono)', 'Ls-23K (mono)', 'Lm-CTC1034 (mono)', 'Ls-23K + Lm-CTC1034 (co)', 'Ls-CTC494 + Lm-CTC1034 (co)']
    mrkrs = ['^', '^', '^', 'o', 'o']
    lst = ['dashed','dashed', 'dashed', 'solid', 'solid']
    clrs = [colors_all['R'], colors_all['N'], colors_all['N_tempshift_10_5_15'], colors_all['N_wo_tempshift'], colors_all['N_lambd_1e-3_omega_0_5']]
    fig, ax = plt.subplots()
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)

        ax.plot(t_model, obs_model[3], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax.scatter(days, obs_x[i][3], color=clrs[i], marker=mrkrs[i])

        #ax2.plot(t_model, obs_model[4], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax2.scatter(days, obs_x[i][4], color=clrs[i], marker='x')
    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Lactic Acid [g/L]')
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', 'pH')


    ax2.set_ylim(-1, 7.)
    ax.set_ylim(-1, 7.)
    legend_elements = [
        Line2D([0], [0], color=clrs[i], label=lbls[i], marker=mrkrs[i], linestyle=lst[i])
        for i in exp_indexes]
    #legend_box = [0.48, 0.75]
    #plt.legend(handles=legend_elements, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)
    plt.legend(handles=legend_elements, loc='center left')
    ax.text(*coord_text, r'\textbf{C}', transform = ax.transAxes)
    
    plt.savefig(path + f"Figures-pool_model-real_data-LA-Ls23.png", bbox_inches="tight")
    plt.close(fig)


        # Plotting lactic acid:
        # For cocultures
    exp_indexes = [0, 2, 4]
    lbls = ['Ls-CTC494 (mono)', 'Ls-23K (mono)', 'Lm-CTC1034 (mono)', 'Ls-23K + Lm-CTC1034 (co)', 'Ls-CTC494 + Lm-CTC1034 (co)']
    mrkrs = ['^', '^', '^', 'o', 'o']
    lst = ['dashed','dashed', 'dashed', 'solid', 'solid']
    clrs = [colors_all['R'], colors_all['N'], colors_all['N_tempshift_10_5_15'], colors_all['N_wo_tempshift'], colors_all['N_lambd_1e-3_omega_0_5']]
    fig, ax = plt.subplots()
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    for i in exp_indexes:
        x0 = set_initial_vals(np.array(x0_vals[n_cl*i:n_cl*(i+1)]), None, n_cl, pH0=obs_x[0][-1][0])
        pH_series = np.array([obs_x[i][-1], days]).T
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode_new, x0, [pH_series, n_cl])
        else:
            x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t_model, param_ode, x0, [pH_series, n_cl])
        obs_model = observable(t_model, x_sol)

        ax.plot(t_model, obs_model[3], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax.scatter(days, obs_x[i][3], color=clrs[i], marker=mrkrs[i])

        #ax2.plot(t_model, obs_model[4], label=lbls[i], linewidth=3.5, color=clrs[i], linestyle=lst[i])
        ax2.scatter(days, obs_x[i][4], color=clrs[i], marker='x')
    ax.set_xlim(-0.1, np.max(t_model))
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Lactic Acid [g/L]')
    fig, ax2 = set_labels(fig, ax2, r'Time, $t$ [h]', 'pH')


    ax2.set_ylim(-1, 7.)
    ax.set_ylim(-1, 7.)
    legend_elements = [
        Line2D([0], [0], color=clrs[i], label=lbls[i], marker=mrkrs[i], linestyle=lst[i])
        for i in exp_indexes]
    #legend_box = [0.43, 0.7]
    #plt.legend(handles=legend_elements, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)
    plt.legend(handles=legend_elements, loc='center left')
    ax.text(*coord_text, r'\textbf{D}', transform = ax.transAxes)
    plt.savefig(path + f"Figures-pool_model-real_data-LA-LsCTC494.png", bbox_inches="tight")
    plt.close(fig)
    return 1


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
            param_ode_new[2*3 + 2 + 3+1] = 0.
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode_new, x_max[i])
        else:
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    print(np.shape(ll_x), np.sum(ll_x, axis=(0)))
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
    #ll_x0 = (obs_x[i][:-1] - C[:-1]) ** 2 / x_max[:-1]
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
    relnoise = 0.1

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/' 
    add_name = ''

    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)


    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    #x0_vals = param_opt[:n_cl] 


    names = ['LsCTC494', 'Ls23K', 'Lm', 'Ls23K-Lm', 'LsCTC494-Lm']
    n_exps = len(names)
    temps = [2.0 for _ in range(len(names))]

    df_names = [f'dataframe_poolpaper_{name}.pkl' for name in names]
    #df_name = f'dataframe_poolpaper.pkl'
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))


    plot_paper_figures(param_opt, dfs, path=path2, add_name=add_name)


    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    param_ode_new = np.copy(param_ode)
    for i in range (len(data)):
        #if exps[i] != 'LsCTC494' and exps[i] != 'LsCTC494-Lm' and exps[i] != 'V01' and exps[i] != 'V05':
        if exps[i] != 'LsCTC494-Lm' and exps[i] != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode_new[2*3 + 2 + 3+1] = 0.
            plot_all_curves(param_ode_new, x0_vals[n_cl*i:n_cl*(i+1)], data=data[i], path=path2, add_name=f'_estim_realdata_{names[i]}')
        else:
            plot_all_curves(param_ode, x0_vals[n_cl*i:n_cl*(i+1)], data=data[i], path=path2, add_name=f'_estim_realdata_{names[i]}')

    calibr_setup = {
        "model": ode_model_coculture,
        "output_path": path2,
        "n_cl": n_cl,
        "dfs": [dfs],
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps,
        "exp_temps": {exp: temp for exp, temp in zip(exps, temps)},
    }
    data_array = extract_observables_from_df(calibr_setup["dfs"])
    calibr_setup["data_array"] = data_array
    cost_res(param_opt, calibr_setup)
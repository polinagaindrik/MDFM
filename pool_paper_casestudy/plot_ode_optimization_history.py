import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ode_model_coculture(t, x, param, x0, ode_args):
    (pH_cond, n_cl,) = ode_args
    (mu_ls, mu_lm, omega_ls_exp, omega_lm_exp, omegaT_lm_exp, N_texp, k_T_exp, k_LA_ls_exp, k_LA_lm_exp, ) = param
    # TODO add pH dependence of mu
    (x_ls0, x_lm0, R0, T0, LA0, pH0) = x0
    (x_ls, x_lm, R, T, LA, pH) = x
    N_t = 10**N_texp
    omega_ls = 10**omega_ls_exp
    omega_lm = 10**omega_lm_exp
    omegaT_lm = 10**omegaT_lm_exp
    k_T = 10**k_T_exp
    k_LA_ls = 10**k_LA_ls_exp
    k_LA_lm = 10**k_LA_lm_exp
    return [
        (mu_ls * R - omega_ls) * x_ls,
        (mu_lm * R - omega_lm) * x_lm - omegaT_lm / (N_t) * T * x_lm,
        -(mu_ls / N_t) * R * x_ls - (mu_lm / N_t) * R * x_lm,
        k_T * x_ls * R,  #  ??
        k_LA_ls * x_ls + k_LA_lm * x_lm,  # *R but wo R the curves look better
        0. # TODO pH evolution with time
    ]


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

def set_initial_vals(x10, temps, n_cl):
    return [[10**L0 for L0 in x10[i]] + [1., 0., 0., 6.] for i in range (len(temps))]

def plot_all_curves(t, param_ode, x10, data=None, path='', add_name=''):
    if data is not None:
        days, [obs_x] = extract_observables_from_df([data])
    x0 = set_initial_vals(x10, temps, n_cl)[0]
    x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t, param_ode, x0, [temps, n_cl])
    lbls = ["ls", "lm", "R", "BAC", "LA", "pH"]
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(t, x_sol[i], label=lbls[i])
        if data is not None:
            ax.scatter(days, obs_x[0][i], label=lbls[i]+'_data', marker='x')
    # ax.plot(t, x_sol[2], label='R')
    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path + f"x_sol_R{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, x_sol[3], label=lbls[3])
    if data is not None:
        ax.scatter(days, obs_x[0][3], label=lbls[3]+'_data', marker='x')
    plt.legend()
    plt.savefig(path + f"BAC{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, x_sol[4], label=lbls[4])
    ax.plot(t, x_sol[5], label=lbls[5])
    if data is not None:
        ax.scatter(days, obs_x[0][4], label=lbls[4]+'_data', marker='x')
        ax.scatter(days, obs_x[0][5], label=lbls[5]+'_data', marker='x')
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    n_cl = 2
    relnoise = 0.1
    n_exps = 1

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/' 
    add_name = ''

    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)


    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    x0_vals = param_opt[:n_cl]
    t_test = np.linspace(0.0, 55.0, 100)
    temps = [2.0 for _ in range(n_exps)]

    names = ['LsCTC494', 'Ls23K', 'Lm', 'LsCTC494_Lm', 'Ls23K_Lm']

    #df_name = f'dataframe_poolpaper_{names[3]}.pkl'
    df_name = f'dataframe_poolpaper.pkl'
    data = pd.read_pickle(path2+df_name)

    plot_all_curves(t_test, param_opt[n_cl*len(temps):], [param_opt[:n_cl*len(temps)]], data=data, path=path2, add_name='_estim_realdata')

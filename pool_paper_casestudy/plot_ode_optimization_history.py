import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ode_model_coculture(t, x, param, x0, ode_args):
    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    (mu_ls_opt, mu_lm_opt, pH_ls_min, pH_ls_opt, pH_lm_min, pH_lm_opt,
    #(mu_ls_opt, mu_lm_opt, pH_ls_min, pH_lm_min,
    omega_ls_exp, omega_lm_exp, omegaT_lm_exp, N_texp, k_T_0, k_LA_ls_exp, k_LA_lm_exp, ) = param
    (x_ls0, x_lm0, R0, T0, LA0, pH0) = x0
    (x_ls, x_lm, R, T, LA, pH) = x

    mu_ls = mu_ls_opt * (pH - pH_ls_min) / (pH_ls_opt - pH_ls_min)
    mu_lm = mu_lm_opt * (pH - pH_lm_min) / (pH_lm_opt - pH_lm_min)
    #mu_ls = mu_ls_opt**2 * (pH - pH_ls_min)**2
    #mu_lm = mu_lm_opt**2 * (pH - pH_lm_min)**2

    N_t = 10**N_texp
    omega_ls = 10**omega_ls_exp
    omega_lm = 10**omega_lm_exp
    omegaT_lm = 10**omegaT_lm_exp
    k_T = 10**(-5) * k_T_0
    k_LA_ls = 10**k_LA_ls_exp
    k_LA_lm = 10**k_LA_lm_exp

    return [
        (mu_ls * R - omega_ls) * x_ls,
        (mu_lm * R - omega_lm) * x_lm - omegaT_lm / (N_t) * T * x_lm,
        -(mu_ls / N_t) * R * x_ls - (mu_lm / N_t) * R * x_lm,
        k_T * x_ls * R,  #  ??
        k_LA_ls * x_ls+ k_LA_lm * x_lm,  # *R but wo R the curves look better
        0. # TODO pH evolution with time
    ]

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


def plot_all_curves(t, param_ode, x10, data=None, path='', add_name=''):
    if data is not None:
        days, [obs_x] = extract_observables_from_df([data])
    x0 = [x10[0], x10[1], 1., 0., 0., 6.]
    pH_series = np.array([obs_x[0][-1], days]).T
    x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t, param_ode, x0, [pH_series, n_cl])
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

def cost_res(param, calibr_setup):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    param_ode = list(param[n_cl*n_exps:])
    x0_vals = param[:n_cl*n_exps]

    (df_x, ) = calibr_setup["dfs"]
    _, [obs_x] = calibr_setup["data_array"]
    n_cl = calibr_setup['n_cl']  # np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    # TODO not clear, should just we compare logaritms?
    x_max = obs_x**2
    x_max[x_max <= 0.1] = 1.0
    #ll_x = np.zeros(np.shape(obs_x))
    ll_x = np.zeros(np.shape(obs_x[:, :-1]))
    for i, exp in enumerate(exps):
        if exp != 'LsCTC494' or exp != 'LsCTC494-Lm' or exp != 'V01' or exp != 'V04':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode[n_cl*3 + n_cl + 2] = 0.
        ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return 1

def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, x_max):
    # TODO mb: do we need to fit also data for BAC, LA (pH)
    # Then obs_x -> obs_x+m
    # + pH instead of temp?
    model = calibr_setup["model"]
    days, [obs_x] = calibr_setup["data_array"]
    temp = calibr_setup["exp_temps"][exp]
    pH_series = np.array([obs_x[i][-1], days]).T
    const = [pH_series, n_cl]

    C0 = np.concatenate((np.array(x0), np.array([1., 0., 0., 6.])))
    C = fm.mdl.model_ODE_solution(model, days, param_ode, C0, const)
    #ll_x0 = (obs_x[i][:-1] - C[:-1]) ** 2 / x_max[:-1]
    ll_x0 = [
        (obs_x[i][0] - C[0]) ** 2 / x_max[0],
        (obs_x[i][1] - C[1]) ** 2 / x_max[1],
        (obs_x[i][2] - C[2]) ** 2,
        (obs_x[i][3] - C[3]) ** 2,
        (obs_x[i][4] - C[4]) ** 2,
    ]
    print(i, ll_x0)
    return np.array(ll_x0)

if __name__ == "__main__":
    n_cl = 2
    relnoise = 0.1

    path = 'out/'
    path2 = 'pool_paper_casestudy/out/test/' 
    add_name = ''

    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    fm.plotting.plot_cost_function(df_optim2, path=path2)


    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    #x0_vals = param_opt[:n_cl]
    t_test = np.linspace(0.0, 55.0, 100)


    names = ['LsCTC494', 'Ls23K', 'Lm', 'LsCTC494_Lm', 'Ls23K_Lm']
    n_exps = len(names)
    temps = [2.0 for _ in range(len(names))]

    df_names = [f'dataframe_poolpaper_{name}.pkl' for name in names]
    #df_name = f'dataframe_poolpaper.pkl'
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))

    x0_vals = param_opt[:n_cl*n_exps]
    param_ode = list(param_opt[n_cl*n_exps:])
    for i in range (len(data)):
        if exps[i] != 'LsCTC494' or exps[i] != 'LsCTC494-Lm' or exps[i] != 'V01' or exps[i] != 'V04':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            param_ode[n_cl*3 + n_cl + 2] = 0.
        plot_all_curves(t_test, param_ode, x0_vals[n_cl*i:n_cl*(i+1)], data=data[i], path=path2, add_name=f'_estim_realdata_{names[i]}')
    
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
    #cost_res(param_opt, calibr_setup)
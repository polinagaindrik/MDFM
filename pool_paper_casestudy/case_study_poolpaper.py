import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import matplotlib.pyplot as plt


def extract_observables_from_df(dfs):
    (df_x,) = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split("_")[3]) for f in df_x.columns]))
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days_x)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days_x):
            df0 = df_x.filter(like=exp).filter(like=f"_{int(d):02d}_")
            if np.shape(df0)[-1] != 0.0:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan * np.ones((np.shape(df_x)[0]))
    return days_x, [obs_x]


def calculate_model_params(cost_func, calibr_setup):
    output_file = "out/optimization_history1.csv"
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup["param_bnds"])):
            output += f"p{i},"
        f.write(output + "cost\n")
    data_array = extract_observables_from_df(calibr_setup["dfs"])
    calibr_setup["data_array"] = data_array
    optim_output = fm.pest.optimization_func(
        cost_func,
        calibr_setup["param_bnds"],
        args=(calibr_setup, None),
        workers=calibr_setup["workers"],
    )
    return np.array(optim_output.x), optim_output.fun


def cost(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    lambd = param[n_cl*n_exps : n_cl*n_exps + n_cl]
    alph = param[n_cl*n_exps + n_cl : n_cl*n_exps + n_cl + n_cl]
    rest_ode_param = param[n_cl*n_exps + n_cl + n_cl:]
    x0_vals = param[:n_cl*n_exps]

    (df_x) = calibr_setup["dfs"]
    _, [obs_x] = calibr_setup["data_array"]
    n_cl = np.shape(obs_x)[1]  # np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    # TODO not clear, should just we compare logaritms?
    x_max = obs_x**2
    ll_x = np.zeros(np.shape(obs_x))
    for i, exp in enumerate(exps):
        param_ode = np.concatenate((lambd, alph, rest_ode_param))
        ll_x[i] = sq_diff_oneexp(
            calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return calibr_setup["aggregation_func"]([ll_x])


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, x_max):
    # TODO mb: do we need to fit also data for BAC, LA (pH)
    # Then obs_x -> obs_x+m
    # + pH instead of temp?
    model = calibr_setup["model"]
    days, [obs_x] = calibr_setup["data_array"]
    temp = calibr_setup["exp_temps"][exp]
    const = [[temp], n_cl, calibr_setup["media"]]

    C0 = np.concatenate((10 ** np.array(x0), np.array([1., 0., 0.])))
    C = fm.mdl.model_ODE_solution(model, days, param_ode, C0, const)
    n_C = C[:2]
    ll_x0 = (obs_x[i] - n_C) ** 2 / x_max
    return ll_x0


def data_calibration(dfs, path=""):
    exps_calibr = sorted(list(set([s.split("_")[0] for s in dfs[0].columns])))
    calibr_presetup = {
        "model": ode_model_coculture,
        "workers": workers,  # number of threads for multiprocessing
        "output_path": path,
        "n_cl": n_cl,
        "dfs": dfs,
        "aggregation_func": fm.pest.cost_arithmetic_mean,
        "exps": exps_calibr,
        "exp_temps": {exp: temp for exp, temp in zip(exps_calibr, temps)},
    }
    fm.output.json_dump(calibr_presetup["exp_temps"], "exp_temps_model_paper.json", dir=path)

    # TODO Change  model so that bounds are on the same scale
    x0_bnds_all = tuple([(2., 6.) for _ in range(calibr_presetup["n_cl"])])
    param_ode_bnds = tuple(
        [(0.01, 1.) for _ in range(n_cl) for _ in range (len(exps_calibr))] +   # mu
        [(0., 1.) for _ in range(n_cl)]   +   # omega
        [(1., 10**6)] + # omegaT
        [(7.0, 10.0)] + # N_max
        [(10**-6, 2*10**-5)] + # k_T
        [(10**-10, 10**-8) for _ in range(n_cl)] # k_LA
    )
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = x0_bnds_all + param_ode_bnds

    print("Start optimization...")
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    fm.output.json_dump({"param_ode": param_opt.astype(list)}, "Result_calibration.json", dir=path)
    return param_opt, calibr_setup


def ode_model_coculture(t, x, param, x0, ode_args):
    (temp_cond, n_cl,) = ode_args
    (mu_ls, mu_lm, omega_ls, omega_lm, omegaT_lm, N_texp, k_T, k_LA_ls, k_LA_lm, ) = param
    # TODO add pH dependence of mu
    (x_ls0, x_lm0, R0, T0, LA0) = x0
    (x_ls, x_lm, R, T, LA) = x
    N_t = 10**N_texp
    return [
        (mu_ls * R - omega_ls) * x_ls,
        (mu_lm * R) * x_lm - omega_lm * x_lm - omegaT_lm / (N_t) * T * x_lm,
        -(mu_ls / N_t) * R * x_ls - (mu_lm / N_t) * R * x_lm,
        k_T * x_ls * R,  #  ??
        k_LA_ls * x_ls + k_LA_lm * x_lm,  # *R but wo R the curves look better
    ]

def ode_model_monoculture(t, x, param, x0, ode_args):
    (temp_cond, n_cl,) = ode_args
    (mu, omega, omegaT, N_t, k_T, k_LA, ) = param
    # If it's lm: k_T=0
    # If it's ls: omegaT=0
    # TODO add pH dependence of mu
    (x0, R0, T0, LA0) = x0
    (x, R, T, LA) = x
    return [
        (mu * R) * x - omega * x - omegaT / (N_t) * T * x,
        -(mu / N_t) * R * x,
        k_T * x * R,  #  ??
        k_LA * x
    ]


if __name__ == "__main__":
    path = "pool_paper_casestudy/out/"
    workers = -1
    n_cl = 2
    # relnoise = 0.1
    n_exps = 1
    add_name = ""
    temps = [2.0 for _ in range(n_exps)]
    ntr = 1
    path_new = path + "test/"

    dfs_calibr = 1
    # dfs_calibr = read_exp_data()
    # param_opt, calibr_setup = data_calibration(dfs_calibr, path=path_new)

    ## Test the model results:
    t_test = np.linspace(0.0, 55.0, 100)  # hours
    x0_test = np.array([10**5., 10**3.1, 1.0, 0.0, 0.0])
    param_ode_test = [
            0.38,           # mu_ls
            0.32,           # mu_lm
            0.001,          # omega_ls
            0.001,          # omega_lm
            10**4,          # omegaT_lm
            8.3,            # N_texp
            10**-5,         # k_T
            10**-9,         # k_LA_ls
            0.5 * 10**-9,   # k_LA_lm
    ]
    x_sol = fm.mdl.model_ODE_solution(
        ode_model_coculture, t_test, param_ode_test, x0_test, [temps, n_cl]
    )

    lbls = ["ls", "lm", "R", "BAC", "LA"]
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(t_test, x_sol[i], label=lbls[i])
    # ax.plot(t_test, x_sol[2], label='R')
    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path_new + "x_sol_R.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t_test, x_sol[-2], label=lbls[-2])
    plt.legend()
    plt.savefig(path_new + "BAC.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t_test, x_sol[-1], label=lbls[-1])
    plt.legend()
    plt.savefig(path_new + "LA.png", bbox_inches="tight")
    plt.close(fig)
import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from math import floor

########### In-silico data generation ############
def data_generation_poolpaper(n_cl, param_ode, x10, path=''):
    dfs_ode = []
    add_name = ''
    temps = [2.,]
    ntr = 1
    df_ode = model_wotemp(n_cl, temps, ntr, param_ode=param_ode, x10=x10, add_name=add_name,path=path, exp_start_offset=0)
    dfs_ode.append(df_ode)
    return df_ode

def model_wotemp(n_cl, temps, ntr, param_ode=None, x10=None, path='',add_name='', exp_start_offset=0):
    np.random.seed(46987)
    t = np.linspace(0., 17., 18)
    if param_ode is None:
       print('No parameter vector provided.')
       exit()
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    df_ode = generate_data_dfs(ode_model_coculture, t, np.array(param_ode), x0, temps, n_cl, n_traj=ntr, exp_start_offset=exp_start_offset)
    fm.data.save_all_dfs([df_ode], names=[f'poolpaper{add_name}'], path=path)
    print(add_name, param_ode, '\n')
    fm.data.json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode)}, f'Generated_param{add_name}.json', dir=path)
    return df_ode

def set_initial_vals(x10, temps, n_cl):
    return [[10**L0 for L0 in x10[i]] + [1., 0., 0., 6.] for i in range (len(temps))]


def generate_data_dfs(model, t, param, x0, temps, n_cl, n_traj=1, exp_start_offset=0):
    df_ode = []
    for j, temp in enumerate(temps):
        exp_start = exp_start_offset + 1 + j
        const = [[temp], n_cl]
        x0_exp = np.asarray(x0[j], dtype=float)
        param_ode = np.asarray(param[:n_cl*(4+n_cl)+2])
        x = fm.mdl.model_ODE_solution(model, t, param_ode, x0_exp, const)#, jac=jac)
        bacteria_name = ['Ls', 'Lm']
        df_ode0 = fm.dtf.merge_dfs([create_df_poolpaper(t, x, [f'V{j+exp_start:02d}'], bacteria_name, stds=0.) for j in range(n_traj)], sort=False)
        df_ode.append(df_ode0)
    return fm.dtf.merge_dfs(df_ode, sort=False)

def create_df_poolpaper(days, obs, name_part, bact_name, stds=0):
    n_cl = len(bact_name)
    n_states = 1
    data = {"Measurement": ['x_'+bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]+['m_Resource', 'm_BAC', 'm_LA', 'pH']}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'poolpaper'])] = o
    df = pd.DataFrame(data=data).set_index('Measurement') 
    #df = pd.DataFrame(data=data).groupby('Measurement', sort=False).sum()
    return df

############### Read data from excel ######################3
# TODO pack this all in df: how to deal with missing times for LA and BAC
def experimental_values(name, skiprows=0, LA_sheetname=''):
    filename = 'CCD_results_counts_Part 2.xlsx'
    df_counts = pd.read_excel("pool_paper_casestudy/data/" + filename, keep_default_na=True, sheet_name='R9_rep', skiprows=skiprows, usecols='A:F', nrows=16)
    # TODO: temporal solution to round all t to the round number: maybe not accurate: what else to do?
    time_count = round(df_counts['Time'])
   
    Ls = np.array(df_counts['LAB (cfu/mL)'])
    Lm = np.array(df_counts['LM (cfu/mL)'])
    pH = np.array(df_counts['pH'])

    if name == 'Ls23K' or 'Lm' or 'Ls23K_Lm':
        time_BAC = time_count
        BAC = np.array([0. for t in time_BAC])
    else:
        df_BAC = pd.read_excel("pool_paper_casestudy/data/8_BA/BA_09.xlsx", keep_default_na=True, sheet_name='BA_prod', skiprows=19, usecols='M:Q', nrows=16)
        time_BAC = round(df_BAC['Time, h (1)'])
        BAC = np.array(df_BAC['BA (10^3 AU/mL)'])*10**3

    df_LA = pd.read_excel("pool_paper_casestudy/data/7_LA/RUN_09.xlsx", keep_default_na=True, sheet_name=LA_sheetname, skiprows=19, usecols='M:Q', nrows=16)
    time_LA = round(df_LA['Time, h (1)'])
    # Restore missing measurements of LA
    time_all = sorted(set(list(time_count) + list(time_BAC) + list(time_LA)))
    df_LA_new = df_LA.T
    j = 0
    for t in time_all:
        if not np.any(np.abs(df_LA['Time, h (1)'] - t) <= 0.3):
            j = j+1
            df_LA_new[len(df_LA['Time, h (1)'])+j] = [t]+ [np.nan for _ in range (len(df_LA.columns)-1)]
    df_LA = df_LA_new.T.sort_values(by=['Time, h (1)'], ascending=True)
    LA = np.array(df_LA['LA (mg/mL)'])

    Resource = [np.nan for _ in range (len(time_all))]
    print(time_all)
    print(len(Ls), len(Lm), len(Resource), len(BAC), len(LA), len(pH))
    print(df_LA)
    obs = np.array([Ls, Lm, Resource, BAC, LA, pH])
    exp_start = 1

    return create_df_poolpaper(time_all, obs, [f'V{exp_start:02d}'], ['Ls', 'Lm'])

###############################################################################################33

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

    (df_x, ) = calibr_setup["dfs"]
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
    const = [[temp], n_cl]

    C0 = np.concatenate((10 ** np.array(x0), np.array([1., 0., 0., 6.])))
    C = fm.mdl.model_ODE_solution(model, days, param_ode, C0, const)
    ll_x0 = (obs_x[i] - C) ** 2 / x_max
    return ll_x0


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
        "exp_temps": {exp: temp for exp, temp in zip(exps_calibr, temps)},
    }
    fm.output.json_dump(calibr_presetup["exp_temps"], "exp_temps_model_paper.json", dir=path)

    # TODO Change  model so that bounds are on the same scale
    # TODO are LA0, BAC0, pH0 also needed to be estimated
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
    (pH_cond, n_cl,) = ode_args
    (mu_ls, mu_lm, omega_ls, omega_lm, omegaT_lm, N_texp, k_T, k_LA_ls, k_LA_lm, ) = param
    # TODO add pH dependence of mu
    (x_ls0, x_lm0, R0, T0, LA0, pH0) = x0
    (x_ls, x_lm, R, T, LA, pH) = x
    N_t = 10**N_texp
    return [
        (mu_ls * R - omega_ls) * x_ls,
        (mu_lm * R) * x_lm - omega_lm * x_lm - omegaT_lm / (N_t) * T * x_lm,
        -(mu_ls / N_t) * R * x_ls - (mu_lm / N_t) * R * x_lm,
        k_T * x_ls * R,  #  ??
        k_LA_ls * x_ls + k_LA_lm * x_lm,  # *R but wo R the curves look better
        0. # TODO pH evolution with time
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

############## Plotting ######################333
def plot_all_curves(t, param_ode, x10, path='', add_name=''):
    x0 = set_initial_vals(x10, temps, n_cl)[0]
    x_sol = fm.mdl.model_ODE_solution(ode_model_coculture, t, param_ode, x0, [temps, n_cl])
    lbls = ["ls", "lm", "R", "BAC", "LA", "pH"]
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(t, x_sol[i], label=lbls[i])
    # ax.plot(t, x_sol[2], label='R')
    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path + f"x_sol_R{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, x_sol[3], label=lbls[3])
    plt.legend()
    plt.savefig(path + f"BAC{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, x_sol[4], label=lbls[4])
    ax.plot(t, x_sol[5], label=lbls[5])
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    path = "pool_paper_casestudy/out/"
    workers = 4
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

    names = ['LsCTC494', 'Ls23K', 'Lm', 'LsCTC494_Lm', 'Ls23K_Lm']
    skip_rows = [8, 34, 58, 83, 109]
    LA_sheetnames = ['R9_494_LA_prod', 'R9_23K_LA_prod', 'R9_1034_LA_prod', 'R9_494co_LA_prod', 'R9_23Kco_LA_prod']

    #for n, nr, las in zip(names, skip_rows, LA_sheetnames):
    #    print(n)
    #    df = experimental_values(n, skiprows=nr, LA_sheetname=las)

    
    ## Test the model results:
    t_test = np.linspace(0.0, 55.0, 100)  # hours
    x0_test = np.array([10**5., 10**3.1, 1.0, 0.0, 0.0, 6.0])
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
    x10_bact = [[5., 3.1]]
    plot_all_curves(t_test, param_ode_test, x10_bact, path=path_new, add_name='_init')

    df_ode = data_generation_poolpaper(n_cl, param_ode_test, x10_bact, path=path_new)
    days_x, [obs_x] = extract_observables_from_df([df_ode])
    param_opt, calibr_setup = data_calibration_poolpaper([df_ode], path=path_new)
    plot_all_curves(t_test, param_opt[n_cl*len(temps):], [param_opt[:n_cl*len(temps)]], path=path_new, add_name='_estim')
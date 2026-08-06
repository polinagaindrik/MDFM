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

########### In-silico data generation ############
def data_generation_poolpaper(n_cl, param_ode, x10, times, path=''):
    dfs_ode = []
    add_name = ''
    temps = [2.,]
    ntr = 1
    df_ode = model_wotemp(n_cl, temps, ntr, times, param_ode=param_ode, x10=x10, add_name=add_name,path=path, exp_start_offset=0)
    dfs_ode.append(df_ode)
    return df_ode

def model_wotemp(n_cl, temps, ntr, times, param_ode=None, x10=None, path='',add_name='', exp_start_offset=0):
    np.random.seed(46987)
    t = times
    if param_ode is None:
       print('No parameter vector provided.')
       exit()
    if x10 is not None:
        x0 = set_initial_vals(x10, None, n_cl)
    df_ode = generate_data_dfs(ode_model_coculture, t, np.array(param_ode), x0, temps, n_cl, n_traj=ntr, exp_start_offset=exp_start_offset)
    fm.data.save_all_dfs([df_ode], names=[f'poolpaper{add_name}'], path=path)
    print(add_name, param_ode, '\n')
    fm.data.json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode)}, f'Generated_param{add_name}.json', dir=path)
    return df_ode

def set_initial_vals(x10, temps, n_cl, pH0=6.):
    return np.concatenate((x10, [1., 0., 0., pH0]))

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
    data = {"Measurement": ['x_'+bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]+['m_BAC', 'm_LA', 'pH']}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'poolpaper'])] = o
    df = pd.DataFrame(data=data).set_index('Measurement') 
    #df = pd.DataFrame(data=data).groupby('Measurement', sort=False).sum()
    return df

############### Read data from excel ######################3
def experimental_values(name, skiprows=0, path_data='', LA_sheetname='', path='', exp_start_offset=0):
    filename = 'CCD_results_counts_Part 2.xlsx'
    df_counts = pd.read_excel(path_data + filename, keep_default_na=True, sheet_name='R9_rep', skiprows=skiprows, usecols='A:F', nrows=16)
    # TODO: temporal solution to round all t to the round number: maybe not accurate: what else to do?
    time_count = df_counts['Time'].astype(int)
   
    Ls = np.array(df_counts['LAB (cfu/mL)'])
    Lm = np.array(df_counts['LM (cfu/mL)'])
    pH = np.array(df_counts['pH'])

    if name == 'Lm':
        Ls = np.array([0. for _ in range(len(Lm))])
    elif name == 'Ls23K' or name == 'LsCTC494':
        Lm = np.array([0. for _ in range(len(Ls))])
    elif name == 'LsCTC494-Lm':
        Lm[-1] = np.nan

    if name == 'Ls23K' or name == 'Lm' or name == 'Ls23K-Lm' or name == 'LsCTC494': # ???
        time_BAC = time_count
        BAC = np.array([0. for t in time_BAC])
    else:
        df_BAC = pd.read_excel(path_data+"/8_BA/BA_09.xlsx", keep_default_na=True, sheet_name='BA_prod', skiprows=19, usecols='M:Q', nrows=16)
        time_BAC = df_BAC['Time, h (1)'].astype(int)
        BAC = np.array(df_BAC['BA (10^3 AU/mL)'])*10**3

    df_LA = pd.read_excel(path_data + "/7_LA/RUN_09.xlsx", keep_default_na=True, sheet_name=LA_sheetname, skiprows=19, usecols='M:Q', nrows=16)
    time_LA = df_LA['Time, h (1)'].astype(int)
    # Restore missing measurements of LA
    time_all = sorted(set(list(time_count) + list(time_BAC) + list(time_LA)))
    df_LA["Time, h (1)"] = df_LA["Time, h (1)"].astype(int)
    df_LA_new = df_LA.T
    j = 0
    for t in time_all:
        if not np.any(np.abs(df_LA['Time, h (1)'] - t) <= 0.3):
            j = j+1
            df_LA_new[len(df_LA['Time, h (1)'])+j] = [t]+ [np.nan for _ in range (len(df_LA.columns)-1)]
    df_LA = df_LA_new.T.sort_values(by=['Time, h (1)'], ascending=True)
    LA = np.array(df_LA['LA (mg/mL)'])
    #LA[LA <= 0.0] = 0.0
    
    #Resource = [np.nan for _ in range (len(time_all))]
    obs = np.array([Ls, Lm, BAC, LA, pH])
    exp_start = exp_start_offset + 1
    df = create_df_poolpaper(time_all, obs, [f'V{exp_start:02d}'], ['Ls', 'Lm'])
    fm.data.save_all_dfs([df], names=['poolpaper_' + name], path=path)
    return df

###############################################################################################

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
    #temp = calibr_setup["exp_temps"][exp]
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
        (obs_x[i][3] - obs_model[3]) ** 2,# / np.max(x_max[3]), #/ x_max[3], # LA
        #(obs_x[i][4] - obs_model[4]) ** 2 / np.max(x_max[4]),  # pH
    ]
    return np.array(ll_x0)

    ###############################################################3
def ode_model_coculture(t, x, param, x0, ode_args):
    (x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, _) = x

    (mu_ls_opt, mu_lm_opt,
    pH_ls_min, pH_ls_opt, pH_ls_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omega_ls_exp, omega_lm_exp,
    omegaT_lm_exp, k_T_inhib0, n,
    N_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp,
    q_acid) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls = mu_ls_opt * (pH - pH_ls_min) * (pH_ls_max - pH) / ((pH_ls_opt - pH_ls_min) * (pH_ls_max - pH_ls_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

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


def ode_model_coculture2(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_ls23K_max,
    pH_lsCTC494_min, pH_lsCTC494_opt, pH_lsCTC494_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omegaT_lm, k_T_inhib, n,
    r_23K,  r_lsCTC494, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls23K = mu_ls23K_opt * (pH - pH_ls23K_min) * (pH_ls23K_max - pH) / ((pH_ls23K_opt - pH_ls23K_min) * (pH_ls23K_max - pH_ls23K_min))
    mu_lsCTC494 = mu_lsCTC494_opt * (pH - pH_lsCTC494_min) * (pH_lsCTC494_max - pH) / ((pH_lsCTC494_opt - pH_lsCTC494_min) * (pH_lsCTC494_max - pH_lsCTC494_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    #print(n, k_T_inhib , T, x_lm_sen)
    toxin_death = omegaT_lm * x_lm_sen * np.abs(T)**n / (k_T_inhib**n + np.abs(T)**n)
    return [
        mu_ls23K * R**r_23K * x_ls23K,
        mu_lsCTC494 * R**r_lsCTC494 * x_lsCTC494,# * (N_ls23K_t /N_lsCTC494_t),
        mu_lm * R * x_lm_sen - toxin_death, #* (N_ls23K_t/N_lm_t)
        mu_lm * R * x_lm_res, # * (N_ls23K_t/N_lm_t),
        #-(mu_ls23K / N_ls23K_t)*R*x_ls23K - (mu_lsCTC494 / N_lsCTC494_t)*R*x_lsCTC494 - (mu_lm / N_lm_t)*R*x_lm_sen - (mu_lm / N_lm_t)*R*x_lm_res,
        -(1/N_lm_t)*(r_23K*mu_ls23K*R**r_23K*x_ls23K + r_lsCTC494*mu_lsCTC494* (R**r_lsCTC494)*x_lsCTC494 + mu_lm*R*(x_lm_sen +
        x_lm_res)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def ode_model_coculture3(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_ls23K_max,
    pH_lsCTC494_min, pH_lsCTC494_opt, pH_lsCTC494_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omegaT_lm, k_T_inhib, n,
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls23K = mu_ls23K_opt * (pH - pH_ls23K_min) * (pH_ls23K_max - pH) / ((pH_ls23K_opt - pH_ls23K_min) * (pH_ls23K_max - pH_ls23K_min))
    mu_lsCTC494 = mu_lsCTC494_opt * (pH - pH_lsCTC494_min) * (pH_lsCTC494_max - pH) / ((pH_lsCTC494_opt - pH_lsCTC494_min) * (pH_lsCTC494_max - pH_lsCTC494_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    toxin_death = omegaT_lm * x_lm_sen * np.abs(T)**n / (k_T_inhib**n + np.abs(T)**n)
    #kappa_LA_lm_2 = 0. # so it does not decompose LA

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t) + mu_lm*(x_lm_sen +
        x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
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

############## Plotting ######################333
def plot_all_curves(param_ode, x10, model=ode_model_coculture, obs=observable, data=None, path='', add_name=''):
    clrs = [colors_all['N_A'], colors_all['N_B'], colors_all['R'], colors_all['T'], colors_all['N']]
    n_cl = 3
    if data is not None:
        days, [obs_x] = extract_observables_from_df([data])
    t = np.linspace(days[0], days[-1], 100)
    x0 = set_initial_vals(x10, None, n_cl, pH0=obs_x[0][-1][0])
    pH_series = np.array([obs_x[0][-1], days]).T
    x_sol = fm.mdl.model_ODE_solution(model, t, param_ode, x0, [pH_series, n_cl])
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

def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    return fig, ax


def get_param_dfs(path, path2):
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    return param_opt, dfs, df_optim2

######
def plot_cases_separately(param_opt, dfs, model, path='', add_name='', exp_indexes=[3, 4 ,0, 1, 2]):
    n_cl = 4
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

    # exp_indexes = [3, 4 ,0, 1, 2]
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
        plt.savefig(path + f"Figures-pool_model_real_data_exp_{names[i]}"+add_name+".png", bbox_inches="tight")
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
        plt.savefig(path + f"Figures-pool_model_real_data_BAC_{names[i]}"+add_name+".png", bbox_inches="tight")
        plt.close(fig)
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import matplotlib.pyplot as plt

def extract_observables_from_df(dfs):
    (df_x, ) = dfs
    exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split('_')[3]) for f in df_x.columns]))
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days_x)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days_x):
            df0 = df_x.filter(like=exp).filter(like=f'_{int(d):02d}_')
            if np.shape(df0)[-1] != 0.:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan*np.ones((np.shape(df_x)[0]))
    return days_x , [obs_x]

def calculate_model_params(cost_func, calibr_setup):
    output_file = 'out/optimization_history1.csv'
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"p{i},"
        f.write(output+"cost\n")
    data_array = extract_observables_from_df(calibr_setup['dfs'])
    calibr_setup['data_array'] = data_array
    optim_output = fm.pest.optimization_func(cost_func, calibr_setup['param_bnds'], args=(calibr_setup, None),
                                            workers=calibr_setup['workers'])
    return np.array(optim_output.x), optim_output.fun


def cost(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps = calibr_setup['exps']
    lambd  = param[n_cl:n_cl+n_cl]
    alph = param[n_cl+n_cl:n_cl+n_cl + n_cl*len(exps)]
    rest_ode_param = param[n_cl+n_cl + n_cl*len(exps):-n_cl*n_media]
    s_x = np.array(param)[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl]

    (df_x) = calibr_setup['dfs']
    _, [obs_x] = calibr_setup['data_array']
    n_cl = np.shape(obs_x)[1] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))
    x_max = obs_x**2
    ll_x = np.zeros(np.shape(obs_x))
    for i, exp in enumerate(exps):
        param_ode = np.concatenate((lambd, alph[n_cl*i:n_cl*(i+1)], rest_ode_param))
        ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals, param_ode, s_x, x_max[i])
    ll_x = ll_x[ll_x!=0]
    return calibr_setup['aggregation_func']([ll_x])


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, s_x, x_max):
    model = calibr_setup['model']
    days, [obs_x] = calibr_setup['data_array']
    temp = calibr_setup['exp_temps'][exp]
    const = [[temp], n_cl, calibr_setup['media']]

    C0 = np.concatenate((10**np.array(x0), np.ones((n_cl+1))))
    C = fm.mdl.model_ODE_solution(model, days, param_ode, C0, const)
    n_C = fm.mdl.get_bacterial_count(C, np.shape(s_x)[-1], 2)        
    ll_x0 = (obs_x[i] - n_C)**2 /x_max
    return ll_x0


def pool_model_2sp_comp_toxin1(t, x, param, x0, const):
    (lambd_A, mu_A, lambd_B, mu_B, N_t, k, omega,) = const
    (L_A0, G_A0, L_B0, G_B0, R0, T0) = x0
    (L_A, G_A, L_B, G_B, R, T) = x

    return [
        - lambd_A * R * L_A,
          lambd_A * R * L_A + (mu_A * R - omega/(N_t) * T) * G_A,
        - lambd_B * R * L_B,
          lambd_B * R * L_B + (mu_B * R) * G_B,
        - (mu_A / N_t) * R * G_A - (mu_B / N_t) * R * G_B,
          k * G_B
        ]

def pool_model_2sp_comp_toxin2(t, x, param, x0, const):
    (lambd_A, mu_A, lambd_B, mu_B, N_t, k, omega,) = const
    (L_A0, G_A0, L_B0, G_B0, R0, T0) = x0
    (L_A, G_A, L_B, G_B, R, T) = x

    return [
        - lambd_A * R * L_A,
          lambd_A * R * L_A + (mu_A * R - omega/(N_t) * T) * G_A,
        - lambd_B * R * L_B,
          lambd_B * R * L_B + (mu_B * R) * G_B,
        - (mu_A / N_t) * R * G_A - (mu_B / N_t) * R * G_B,
          k * G_B  * R # * mu_B
        ]

def ode_model(t, x, param, x0, ode_args):
    (temp_cond, n_cl, media) = ode_args
    (mu_ls, mu_lm, omega_ls, omega_lm, omegaT_lm,  N_t, k_T, k_LA,) = param
    # TODO add pH dependence of mu
    (x_ls0, x_lm0, R0, T0, LA0) = x0
    (x_ls, x_lm, R, T, LA) = x
    return [
          (mu_ls * R - omega_ls ) * x_ls,
          (mu_lm * R) * x_lm - omega_lm - omegaT_lm/(N_t) * T,
        - (mu_ls / N_t) * R * x_ls - (mu_lm / N_t) * R * x_lm,
          k_T * x_lm  , # * R ??
          k_LA * x_lm  * R
        ]

def data_calibration(dfs, path=''):
    (df_maldi) = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs[0].columns])))

    calibr_presetup={
            'model': ode_model,
            'workers': workers, # number of threads for multiprocessing
            'output_path': path,
            'n_cl': n_cl,
            'dfs': dfs,
            'aggregation_func': fm.pest.cost_arithmetic_mean,
            'exps': exps_calibr,
            'exp_temps': {exp: temp for exp, temp in zip(exps_calibr, temps)},
            'media': media, 
        }
    fm.output.json_dump(calibr_presetup['exp_temps'], 'exp_temps_model_paper.json', dir=path)

    # TODO Change depending on the model !!! 
    x0_bnds_all = tuple([(1., 4.5) for _ in range (calibr_presetup['n_cl'])])
    inhib_bnds = [(0.02, 3.) for _ in range (n_cl*(n_cl-1))]
    param_ode_bnds = tuple([(.01,  5.) for _ in range (n_cl) ] + # alph
                            [(6., 12.)]                        + # N_max
                            inhib_bnds)
    calibr_setup = calibr_presetup
    calibr_setup['param_bnds'] = x0_bnds_all + param_ode_bnds

    print('Start optimization...')
    param_opt = calculate_model_params(cost, calibr_setup)[0]
    fm.output.json_dump({'param_ode': param_opt.astype(list)}, 'Result_calibration.json', dir=path)
    return param_opt, calibr_setup


if __name__ == "__main__":
    path = 'pool_paper_casestudy/out/'
    workers = -1
    n_cl = 2
    #relnoise = 0.1
    n_exps = 3
    add_name = ''
    temps = [2. for _ in range (n_exps)]
    ntr = 1
    path_new = path + 'test/'

    dfs_calibr = 1
    # dfs_calibr = read_exp_data()
    param_opt, calibr_setup = data_calibration(dfs_calibr, path=path_new)
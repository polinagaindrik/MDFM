import numpy as np
from scipy.integrate import solve_ivp


def calc_obs_model(data, param_x0, calibr_setup, t_model):
    (df_mibi, df_maldi, df_ngs,) = data
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(data[1])[0]
    param_opt = param_x0[n_cl*len(exps):]
    s_x = calibr_setup['s_x']
    obs_mibi = np.zeros((len(exps), len(s_x), len(t_model)))
    obs_maldi = np.zeros((len(exps), len(s_x), np.shape(df_maldi)[0], len(t_model)))
    obs_ngs = np.zeros((len(exps), np.shape(df_ngs)[0], len(t_model)))
    temps = []
    x_count = np.zeros((len(exps) , np.shape(df_maldi)[0], len(t_model)))
    for i, exp in enumerate(exps):
        #df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        #temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        temp = calibr_setup['exp_temps'][exp]
        const = [[temp], n_cl, calibr_setup['media']]
        #C0_opt = 1e3*np.concatenate((param_x0_opt[n_cl*i:n_cl*(i+1)],[1e-3 for _ in range(n_cl)]+[1e-3])) 
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(calibr_setup['model'], t_model, param_opt, C0_opt, const)
        n_C = get_bacterial_count(C, n_cl, 2)
        x_count[i] = n_C
        n_C0 = get_bacterial_count(np.array(C0_opt).reshape(len(C0_opt), 1), n_cl, 2)

        obs_mibi[i] = observable_MiBi(t_model, n_C, calibr_setup['s_x'], n_C0, const, t_model)[1]
        obs_maldi[i] = observable_MALDI(t_model, n_C, calibr_setup['s_x'], n_C0, const, t_model)[1]
        obs_ngs[i] = observable_NGS(t_model, n_C, calibr_setup['T_x'], n_C0, const, t_model)[1]
        temps.append(temp)
    return x_count, obs_mibi, obs_maldi, obs_ngs, temps


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const, jac=None, jac_spasity=None):
    sol_model = solve_ivp(model, [0., t[-1]], x0, dense_output=False, method='LSODA', max_step=0.1, t_eval=t, args=(param, x0, const),
                          rtol=1e-3, atol=1e-3, jac=jac)#, lband=const[1], uband=2*const[1])#, jac_spasity=jac_spasity
    return sol_model.y


# Get bacterial count array from bacterial states x
def get_bacterial_count(x, n_cl, n_states):
    return np.array([np.sum([x[i+n_cl*j] for j in range (n_states)], axis=0) for i in range (n_cl)])


# Get [fpc(xi), fmrs(xi)]
def media_filtering(t, n, s_x, x0, const):
    (temp_cond, n_cl, media) = const
    f_media = np.array(s_x).reshape(-1, n_cl, 1) 
    x_filt = n.reshape((1,)+np.shape(n))*f_media #(n_media, n_sp, n_times)
    return x_filt


def observable_MiBi(t, n, s_x, x0, const, t_meas, std=0.):
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    f_x = media_filtering(t, n, s_x, x0, const)
    return t_meas, np.sum(f_x, axis=1)


def observable_MiBi_woS(t, n, s_x, x0, const, t_meas, std=0.):
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    return t_meas, np.sum(n, axis=0)


def observable_MALDI(t, n, s_x, x0, const, t_meas, std=0.):
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    f_x = media_filtering(t, n, s_x, x0, const)
    f_x = f_x + np.random.normal(0., std, size=np.shape(f_x)) 
    f_x[f_x<=0.005] = 0
    return t_meas, np.array([f/np.sum(f, axis=0) for f in f_x])


def observable_NGS(t, n, T_x, x0, const, t_meas, std=0.):
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    # NGS filtering with T matrix
    f_x = media_filtering(t, n, T_x, x0, const)[0]
    f_x = f_x + np.random.normal(0., std, size=np.shape(f_x))
    f_x[f_x<0] = 0
    return t_meas, np.array(f_x/np.sum(f_x, axis=0))


def observable_x(t, n, s_x, x0, const, t_meas, std=0.):
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    return t_meas, n
import numpy as np
from scipy.stats.mstats import gmean
from fusion_model.model.solving import model_ODE_solution, get_bacterial_count
from fusion_model.model.solving import observable_NGS, observable_MALDI, observable_MiBi


def squared_differences(param_ode, x0_vals, s_x, calibr_setup, jac_spasity=None):
    (df_mibi, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, [obs_mibi_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    model = calibr_setup['model']
    T_x = calibr_setup['T_x']
    n_cl = np.shape(obs_maldi_meas)[2] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    mibi_max = np.nanmax(obs_mibi_meas, axis=(0, 1))**2
    ll_ngs, ll_maldi, ll_mibi = np.zeros(np.shape(obs_ngs_meas)), np.zeros(np.shape(obs_maldi_meas)), np.zeros(np.shape(obs_mibi_meas))
    for i, exp in enumerate(exps):
        #temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        temp = calibr_setup['exp_temps'][exp]
        const = [[temp], n_cl]

        C0 = np.concatenate((10**np.array(x0_vals[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(model, days, param_ode, C0, const)
        n_C = get_bacterial_count(C, np.shape(s_x)[-1], 2)
        n_C0 = get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(s_x)[-1], 2)

        _, obs_ngs = observable_NGS(days, n_C, T_x, n_C0, const, days)
        ll_ngs[i] = (obs_ngs_meas[i] - obs_ngs)**2

        _, obs_maldi = observable_MALDI(days, n_C, s_x, n_C0, const, days)
        ll_maldi[i] = (obs_maldi_meas[i] - obs_maldi)**2
        
        _, obs_mibi = observable_MiBi(days, n_C, s_x, n_C0, const, days)
        ll_mibi[i] = (obs_mibi_meas[i] - obs_mibi)**2 /mibi_max
    return ll_ngs[ll_ngs!=0], ll_maldi[ll_maldi!=0], ll_mibi[ll_mibi!=0]


def cost_withS(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps = calibr_setup['exps']
    param_ode = param[n_cl*len(exps):-n_cl*n_media]
    s_x = np.array(param)[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func'](ll_ngs, ll_maldi, ll_mibi)


def cost_direct(param, calibr_setup, jac_spasity):
    s_x = calibr_setup['s_x']
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = param[n_cl*len(exps):]
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func'](ll_ngs, ll_maldi, ll_mibi)


def cost_initvals(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func'](ll_ngs, ll_maldi, ll_mibi)


def cost_initvals_lambda(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    lambd = param[n_cl*len(exps):]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(np.concatenate((lambd, param_ode)), x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func'](ll_ngs, ll_maldi, ll_mibi)


def cost_logsumexp(ll_ngs, ll_maldi, ll_mibi):
    a = .1
    num_meas = np.size(ll_ngs[~np.isnan(ll_ngs)]) + np.size(ll_maldi[~np.isnan(ll_maldi)]) + np.size(ll_mibi[~np.isnan(ll_mibi)]) 
    return np.log((np.nansum(np.exp(a*ll_ngs)) + np.nansum(np.exp(a*ll_maldi)) + np.nansum(np.exp(a*ll_mibi)))/num_meas)/a


def cost_geometric_mean(ll_ngs, ll_maldi, ll_mibi):
    num_meas = np.size(ll_ngs[~np.isnan(ll_ngs)]) + np.size(ll_maldi[~np.isnan(ll_maldi)]) + np.size(ll_mibi[~np.isnan(ll_mibi)])
    return np.nanprod(ll_ngs**(1/num_meas))*np.nanprod(ll_maldi**(1/num_meas))*np.nanprod(ll_mibi)**(1/num_meas)


def cost_arithmetic_mean(ll_ngs, ll_maldi, ll_mibi):
    return np.nanmean([np.nanmean(ll_ngs), np.nanmean(ll_maldi), np.nanmean(ll_mibi)])


def cost_sum_and_geometric_mean(ll_ngs, ll_maldi, ll_mibi):
    ngs_term = np.nansum(ll_ngs)
    maldi_term = np.nansum(ll_maldi)
    mibi_term = np.nansum(ll_mibi)
    return gmean([ngs_term, maldi_term, mibi_term], nan_policy='omit', axis=None)
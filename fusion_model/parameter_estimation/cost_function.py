import numpy as np
from scipy.stats.mstats import gmean
from fusion_model.model.solving import model_ODE_solution, get_bacterial_count
from fusion_model.model.solving import observable_NGS, observable_MALDI, observable_MiBi
from fusion_model.parameter_estimation.media_matrix import S_squared_difference


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, s_x, mibi_max):
    """Simulate one experiment and return squared residuals for NGS, MALDI, and MiBi."""
    #temp = float(df_mibi0.columns[0].split("_")[2][:-1])
    model = calibr_setup['model']
    days, [obs_mibi_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    temp = calibr_setup['exp_temps'][exp]
    T_x = calibr_setup['T_x']
    const = [[temp], n_cl, calibr_setup['media']]

    C0 = np.concatenate((10**np.array(x0), np.ones((n_cl+1))))
    C = model_ODE_solution(model, days, param_ode, C0, const)
    n_C = get_bacterial_count(C, np.shape(s_x)[-1], 2)
    n_C0 = get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(s_x)[-1], 2)

    _, obs_ngs = observable_NGS(days, n_C, T_x, n_C0, const, days)
    ll_ngs0 = (obs_ngs_meas[i] - obs_ngs)**2

    _, obs_maldi = observable_MALDI(days, n_C, s_x, n_C0, const, days)
    ll_maldi0 = (obs_maldi_meas[i] - obs_maldi)**2
        
    _, obs_mibi = observable_MiBi(days, n_C, s_x, n_C0, const, days)
    ll_mibi0 = (obs_mibi_meas[i] - obs_mibi)**2 /mibi_max
    return ll_ngs0, ll_maldi0, ll_mibi0


def squared_differences(param_ode, x0_vals, s_x, calibr_setup, jac_spasity=None):
    """Collect the nonzero observation residuals across all calibration experiments."""
    (df_mibi, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, [obs_mibi_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    n_cl = np.shape(obs_maldi_meas)[2] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    mibi_max = np.nanmax(obs_mibi_meas, axis=(0, 1))**2
    ll_ngs, ll_maldi, ll_mibi = np.zeros(np.shape(obs_ngs_meas)), np.zeros(np.shape(obs_maldi_meas)), np.zeros(np.shape(obs_mibi_meas))
    for i, exp in enumerate(exps):
        ll_ngs[i], ll_maldi[i], ll_mibi[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, s_x, mibi_max)
    return ll_ngs[ll_ngs!=0], ll_maldi[ll_maldi!=0], ll_mibi[ll_mibi!=0]


def residuals_withS(param, calibr_setup, jac_spasity=None):
    """
    Signed residuals of cost_withS, for least-squares solvers:
    returns ([r_ngs, r_maldi, r_mibi], [exp_ngs, exp_maldi, exp_mibi]) with
    r_k = measured - model (MiBi divided by its maximum, as in sq_diff_oneexp) for all
    measured (non-NaN) entries, and exp_k = index of the experiment of each residual
    (sorted experiment order, as the x0 block of param). sum(r_k**2) equals the sum of
    the squared differences that cost_withS passes to calibr_setup['aggregation_func'].
    The length of each r_k depends only on the data, not on param.
    """
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps_setup = calibr_setup['exps']
    param = np.asarray(param, dtype=float)
    param_ode = param[n_cl*len(exps_setup):-n_cl*n_media]
    s_x = param[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl*len(exps_setup)]

    (df_mibi, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, meas = calibr_setup['data_array']
    obs_mibi_meas, obs_maldi_meas, obs_ngs_meas = meas
    n_cl_d = np.shape(obs_maldi_meas)[2]
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    mibi_max = np.nanmax(obs_mibi_meas, axis=(0, 1))
    model = calibr_setup['model']
    T_x = calibr_setup['T_x']
    m_ngs, m_maldi, m_mibi = (np.full(np.shape(obs_ngs_meas), np.nan), np.full(np.shape(obs_maldi_meas), np.nan),
                              np.full(np.shape(obs_mibi_meas), np.nan))
    for i, exp in enumerate(exps):
        const = [[calibr_setup['exp_temps'][exp]], n_cl_d, calibr_setup['media']]
        C0 = np.concatenate((10**np.array(x0_vals[n_cl_d*i:n_cl_d*(i+1)]), np.ones((n_cl_d+1))))
        C = model_ODE_solution(model, days, param_ode, C0, const)
        n_C = get_bacterial_count(C, np.shape(s_x)[-1], 2)
        n_C0 = get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(s_x)[-1], 2)
        m_ngs[i] = observable_NGS(days, n_C, T_x, n_C0, const, days)[1]
        m_maldi[i] = observable_MALDI(days, n_C, s_x, n_C0, const, days)[1]
        m_mibi[i] = observable_MiBi(days, n_C, s_x, n_C0, const, days)[1]

    res, owner = [], []
    for obs, mdl_obs, scale in ((obs_ngs_meas, m_ngs, 1.), (obs_maldi_meas, m_maldi, 1.),
                                (obs_mibi_meas, m_mibi, mibi_max)):
        obs = np.asarray(obs, dtype=float)
        mask = ~np.isnan(obs)
        r = (obs - mdl_obs) / scale
        res.append(r[mask])
        owner.append(np.broadcast_to(np.arange(len(obs)).reshape((-1,) + (1,) * (obs.ndim - 1)),
                                     obs.shape)[mask])
    return res, owner


def cost_withS_withdiffalpha(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost with experiment-specific alpha parameters and fitted media weights."""
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps = calibr_setup['exps']
    lambd  = param[n_cl*len(exps):n_cl*len(exps)+n_cl]
    alph = param[n_cl*len(exps)+n_cl:n_cl*len(exps)+n_cl + n_cl*len(exps)]
    rest_ode_param = param[n_cl*len(exps)+n_cl + n_cl*len(exps):-n_cl*n_media]
    s_x = np.array(param)[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl*len(exps)]

    (df_mibi, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, [obs_mibi_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    n_cl = np.shape(obs_maldi_meas)[2] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    mibi_max = np.nanmax(obs_mibi_meas, axis=(0, 1))**2
    ll_ngs, ll_maldi, ll_mibi = np.zeros(np.shape(obs_ngs_meas)), np.zeros(np.shape(obs_maldi_meas)), np.zeros(np.shape(obs_mibi_meas))
    for i, exp in enumerate(exps):
        param_ode = np.concatenate((lambd, alph[n_cl*i:n_cl*(i+1)], rest_ode_param))
        ll_ngs[i], ll_maldi[i], ll_mibi[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, s_x, mibi_max)
    ll_ngs = ll_ngs[ll_ngs!=0]
    ll_maldi = ll_maldi[ll_maldi!=0]
    ll_mibi = ll_mibi[ll_mibi!=0]
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])


def cost_withS(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost with ODE parameters and media weights fitted together."""
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps = calibr_setup['exps']
    # The optimization vector is [initial values per experiment, ODE parameters,
    # flattened media-by-cluster filtering weights].
    param_ode = param[n_cl*len(exps):-n_cl*n_media]
    s_x = np.array(param)[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    #ll_s1, ll_s2 = S_squared_difference(np.concatenate((param[-n_cl*n_media:], [1.])), calibr_setup['dfs'], calibr_setup['data_array'], calibr_setup['T_x'])
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])#, ll_s1, ll_s2])


def cost_direct(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost using media weights fixed in the setup."""
    s_x = calibr_setup['s_x']
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = param[n_cl*len(exps):]
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])


def cost_initvals(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost while fitting experiment-specific initial values only."""
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])


def cost_initvals_lambda(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost while fitting initial values and lambda parameters."""
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    lambd = param[n_cl*len(exps):]
    ll_ngs, ll_maldi, ll_mibi = squared_differences(np.concatenate((lambd, param_ode)), x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])


def cost_logsumexp(J_vect):#ll_ngs, ll_maldi, ll_mibi):
    """Combine residual arrays with a smooth log-sum-exp aggregate, ignoring NaNs."""
    a = .1
    num_meas = np.sum([np.size(Ji[~np.isnan(Ji)]) for Ji in J_vect])
    return np.log(np.nansum([np.nansum(np.exp(a*Ji)) for Ji in J_vect])/num_meas)/a


def cost_geometric_mean(J_vect):#ll_ngs, ll_maldi, ll_mibi):
    """Compute the geometric mean across the supplied residual values, ignoring NaNs."""
    num_meas = np.sum([np.size(Ji[~np.isnan(Ji)]) for Ji in J_vect])
    return np.nanprod([Ji**(1/num_meas) for Ji in J_vect])


def cost_arithmetic_mean(J_vect):#ll_ngs, ll_maldi, ll_mibi):
    """Return the mean of each residual array's mean, ignoring NaNs."""
    return np.nanmean([np.nanmean(Ji) for Ji in J_vect])


def cost_sum_and_geometric_mean(J_vect):#ll_ngs, ll_maldi, ll_mibi):
    """Return the geometric mean of the non-NaN residual sum from each array."""
    return gmean([np.nansum(Ji) for Ji in J_vect], nan_policy='omit', axis=None)


DATA_TYPES = ('ngs', 'maldi', 'mibi')  # order of J_vect in cost_withS / squared_differences


class SelectedDataTypes:
    """
    Aggregation over a subset of the data types: calls aggregation_func only with the
    residual arrays of the data types in use (names from DATA_TYPES), e.g.
    SelectedDataTypes(cost_sum_and_geometric_mean, ['ngs', 'maldi']) calibrates without MiBi.
    A class (not a closure) so that calibr_setup stays picklable for the worker pools.
    """
    def __init__(self, aggregation_func, use):
        """Store the wrapped aggregation and the data types in use."""
        unknown = set(use) - set(DATA_TYPES)
        if unknown:
            raise ValueError(f'unknown data types {sorted(unknown)}, choose from {DATA_TYPES}')
        self.aggregation_func = aggregation_func
        self.use = [k in use for k in DATA_TYPES]

    def __call__(self, J_vect):
        """Aggregate the residual arrays of the selected data types."""
        return self.aggregation_func([J for J, u in zip(J_vect, self.use) if u])
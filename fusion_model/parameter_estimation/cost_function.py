from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, polygamma
from scipy.stats.mstats import gmean
from fusion_model.model.solving import model_ODE_solution, get_bacterial_count
from fusion_model.model.solving import observable_NGS, observable_MALDI, observable_MiBi
from fusion_model.parameter_estimation.media_matrix import S_squared_difference


# ----------------------------------------------------------------------
# Noise model of the residuals: calibr_setup['noise_model']
#   'absolute' (default): residual = measured - model (MiBi divided by its maximum)
#   'relative': for multiplicative noise, measured = model * (1 + eps):
#       MiBi (counts):          log(measured) - log(model)
#       MALDI, NGS (fractions): centred log-ratio residual within each composition (over the species of one
#                               experiment / medium / time point): log-ratios are not affected by the
#                               normalization to proportions, the noise becomes eps_i - mean(eps)
#   Entries with measured <= 0 (below detection, absent species) are not used; a composition with fewer than
#   2 present species carries no information. Each composition has (present species - 1) degrees of freedom
#   (count_data_points_noise), which -2 log L objectives must use as the number of data points.
# ----------------------------------------------------------------------
#
# calibr_setup['maldi_n'] = N (None: off, needs noise_model 'relative' and pl.make_gaussian_neg2loglik(..., maldi_n=N)):
#   MALDI as N identified colonies per plate, counts k_j = N * measured ~ Multinomial(N, p_j(model)). Residuals
#   are the deviance residuals of the counts (multinomial_deviance_residuals), one per species incl. the undetected
#   ones (k_j = 0); their sum of squares is -2 log L of the multinomial up to a constant, with no noise level to estimate.
# calibr_setup['ngs_dm'] = (N, A) (None: off, needs noise_model 'relative' and pl.make_gaussian_neg2loglik(..., ngs_dm=(N, A))):
#   NGS as N reads per sample with species counts k ~ Dirichlet-multinomial(N, A p), including k_j = 0. The residuals
#   are dm_deviance_residuals; their sum of squares is -2 log L up to a constant, with N and A known.
# ----------------------------------------------------------------------
NOISE_MODELS = ('absolute', 'relative')


def _noise_model(calibr_setup):
    nm = calibr_setup.get('noise_model') or 'absolute'
    if nm not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {nm!r}")
    return nm


def maldi_n_of(calibr_setup):
    """Number of identified colonies per plate if MALDI is fitted with the multinomial likelihood, else None;
    checks that the noise model and the objective fit to it."""
    n = calibr_setup.get('maldi_n')
    agg = calibr_setup.get('aggregation_func')
    agg = getattr(agg, 'aggregation_func', agg)  # inside SelectedDataTypes
    is_multinomial = bool(getattr(agg, 'is_deviance', (False, False, False))[1])
    if not n:
        if is_multinomial:
            raise ValueError("aggregation MultinomialMaldiNeg2LogLik needs calibr_setup['maldi_n']")
        return None
    if _noise_model(calibr_setup) != 'relative':
        raise ValueError("calibr_setup['maldi_n'] needs noise_model = 'relative'")
    if not is_multinomial:
        raise ValueError("calibr_setup['maldi_n'] needs the objective pl.make_gaussian_neg2loglik(..., maldi_n=N) "
                         f"('neg2logL'), got {type(agg).__name__ if agg is not None else None}")
    return int(n)


def ngs_dm_of(calibr_setup):
    """(reads per sample N, concentration A) if NGS is fitted with the Dirichlet-multinomial likelihood, else None;
    checks that the noise model and the objective fit to it."""
    dm = calibr_setup.get('ngs_dm')
    agg = calibr_setup.get('aggregation_func')
    agg = getattr(agg, 'aggregation_func', agg)  # inside SelectedDataTypes
    is_dm = bool(getattr(agg, 'is_deviance', (False, False, False))[0])
    if not dm:
        if is_dm:
            raise ValueError("this aggregation has NGS as deviance and needs calibr_setup['ngs_dm']")
        return None
    if _noise_model(calibr_setup) != 'relative':
        raise ValueError("calibr_setup['ngs_dm'] needs noise_model = 'relative'")
    if not is_dm:
        raise ValueError("calibr_setup['ngs_dm'] needs the objective pl.make_gaussian_neg2loglik(..., ngs_dm=(N, A)) "
                         f"('neg2logL'), got {type(agg).__name__ if agg is not None else None}")
    return int(dm[0]), float(dm[1])


def multinomial_counts(meas, n):
    """Counts n * meas of proportions (NaN stays NaN); error if they are not whole numbers, i.e. if the data do
    not come from n identified colonies per plate (MALDI) or n reads per sample (NGS)."""
    k = np.asarray(meas, dtype=float) * n
    if np.nanmax(np.abs(k - np.round(k)), initial=0.) > 1e-6:
        raise ValueError(f'proportions are not multiples of 1/{n}: the data do not have {n} counts per composition')
    return np.round(k)


def multinomial_deviance_residuals(k, mu, xp=np):
    """
    Signed deviance residuals sign(k - mu) * sqrt(2 (k log(k / mu) - (k - mu))) of counts k with expected counts
    mu = N * p, for numpy or jax.numpy (xp). If sum(k) = sum(mu) = N within a composition, their sum of squares is
    2 sum_j k_j log(k_j / mu_j) = -2 log L of the multinomial + constant. Undetected species: k = 0 -> -sqrt(2 mu).
    """
    mu = xp.maximum(mu, 1e-300)
    delta = (k - mu) / mu
    near = xp.abs(delta) < 1e-4  # k = mu: series expansion (sqrt(0) has no derivative)
    series = (k - mu) / xp.sqrt(mu) * (1. - delta / 6.)
    k_log = xp.where(k > 0, k * xp.log(xp.where(k > 0, k, 1.) / mu), 0.)
    d = xp.where(near, 1., xp.maximum(2. * (k_log - (k - mu)), 1e-300))
    return xp.where(near, series, xp.sign(k - mu) * xp.sqrt(d))


def multinomial_residuals(meas, model, n):
    """Deviance residuals of MALDI proportions meas against the model proportions (NaN where meas is NaN)."""
    k = multinomial_counts(meas, n)
    use = np.isfinite(k)
    r = multinomial_deviance_residuals(np.where(use, k, 0.), np.where(use, np.asarray(model, dtype=float), 1.) * n)
    return np.where(use, r, np.nan)


def _dm_rate(n, conc):
    """Lagrange multiplier c = log(1 + N / A) of the Dirichlet-multinomial residuals (dm_deviance_residuals)."""
    return float(np.log1p(n / conc))


def dm_saturated(k, n, conc):
    """
    Per-entry maximum of g_k(a) = log Gamma(k + a) - log Gamma(a) - c a, a = A p, c = _dm_rate(n, conc), for the counts k.
    k = 0: a* = 0, g* = 0. k > 0: g_k is concave, a* solves digamma(k + a) - digamma(a) = c.
    Returns arrays of the shape of k: a*, g*, s2 = sqrt(-g''(a*)) and s3 = g'''(a*) / (6 g''(a*)) (series near a*).
    """
    k = np.asarray(k, dtype=float)
    c = _dm_rate(n, conc)
    values, inverse = np.unique(k, return_inverse=True)
    a_star, g_star, s2, s3 = np.zeros(values.shape), np.zeros(values.shape), np.ones(values.shape), np.zeros(values.shape)
    for i, v in enumerate(values):
        if v <= 0:
            continue
        lo = v / np.expm1(c)  # sum_i 1 / (a + i) >= log(1 + k / a), so a* >= k / (exp(c) - 1)
        lo = lo * (1. - 1e-9)
        hi = 2. * lo + 1.
        root = lambda a: (polygamma(0, v + a) - polygamma(0, a)) - c
        while root(hi) > 0:
            hi *= 2.
        a = brentq(root, lo, hi, xtol=1e-12, rtol=1e-13)
        g2 = polygamma(1, v + a) - polygamma(1, a)
        g3 = polygamma(2, v + a) - polygamma(2, a)
        a_star[i], g_star[i] = a, gammaln(v + a) - gammaln(a) - c * a
        s2[i], s3[i] = np.sqrt(-g2), g3 / (6. * g2)
    return tuple(x[inverse].reshape(k.shape) for x in (a_star, g_star, s2, s3))


@lru_cache(maxsize=32)
def _dm_saturated_cached(k_bytes, shape, n, conc):
    """dm_saturated for the numpy cost: the counts are fixed during a fit, so it is computed once per dataset."""
    return dm_saturated(np.frombuffer(k_bytes).reshape(shape), n, conc)


def dm_deviance_residuals(k, a, sat, c, xp=np, lgamma=gammaln):
    """
    Signed residuals sign(a* - a) sqrt(2 (g_k(a*) - g_k(a))) of counts k with a = A p, sat = dm_saturated(k, N, A),
    c = _dm_rate(N, A), for numpy or jax.numpy (xp, lgamma = its log Gamma). Each square is >= 0 and, if sum(p) = 1 within a
    composition, their sum is -2 log L of the Dirichlet-multinomial + constant: the term c a sums to the constant c A.
    Undetected species: k = 0 -> -sqrt(2 c a).
    """
    a_star, g_star, s2, s3 = sat
    has = k > 0
    a_k = xp.maximum(a, 1e-12)
    a_gamma = xp.where(has, a_k, 1.)
    g = xp.where(has, lgamma(k + a_gamma) - lgamma(a_gamma), 0.) - c * xp.where(has, a_k, xp.maximum(a, 1e-300))
    x = a_k - a_star
    near = has & (xp.abs(x) < 1e-3 * xp.where(has, a_star, 1.))  # a = a*: series expansion (sqrt(0) has no derivative)
    series = -x * s2 * (1. + x * s3)
    d = xp.where(near, 1., xp.maximum(2. * (g_star - g), 1e-300))
    return xp.where(near, series, xp.sign(a_star - a) * xp.sqrt(d))


def dm_residuals(meas, model, n, conc):
    """Deviance residuals of NGS proportions meas against the model proportions (NaN where meas is NaN)."""
    k = multinomial_counts(meas, n)
    use = np.isfinite(k)
    k0 = np.where(use, k, 0.)
    sat = _dm_saturated_cached(k0.tobytes(), k0.shape, n, conc)
    r = dm_deviance_residuals(k0, np.where(use, np.asarray(model, dtype=float), 1.) * conc, sat, _dm_rate(n, conc))
    return np.where(use, r, np.nan)


def relative_mask(meas, kind):
    """Data entries used by the relative noise model (depends only on the data).
    kind: 'counts' (MiBi) or 'composition' (MALDI, NGS; species on axis -2)."""
    meas = np.asarray(meas, dtype=float)
    use = np.isfinite(meas) & (meas > 0)
    if kind == 'composition':
        use = use & (use.sum(axis=-2, keepdims=True) >= 2)
    return use


def relative_residuals(meas, model, kind):
    """Residuals of the relative noise model (NaN where not used), see relative_mask."""
    meas = np.asarray(meas, dtype=float)
    model = np.asarray(model, dtype=float)
    use = relative_mask(meas, kind)
    d = np.log(np.where(use, meas, 1.)) - np.log(np.maximum(np.where(use, model, 1.), 1e-300))
    if kind == 'composition':
        n_present = np.maximum(use.sum(axis=-2, keepdims=True), 1)
        d = d - np.where(use, d, 0.).sum(axis=-2, keepdims=True) / n_present
    return np.where(use, d, np.nan)


def count_data_points_noise(data_array, noise_model='absolute', maldi_n=None, ngs_dm=None):
    """Number of data points (degrees of freedom) per data type [ngs, maldi, mibi] for a noise model.
    maldi_n: MALDI fitted with the multinomial likelihood, (species - 1) per measured composition incl. zeros.
    ngs_dm: the same for NGS fitted with the Dirichlet-multinomial likelihood."""
    _, (obs_mibi, obs_maldi, obs_ngs) = data_array
    if noise_model == 'relative':
        def comp_dof(obs):
            use = relative_mask(obs, 'composition')
            return int(np.sum(np.maximum(use.sum(axis=-2) - 1, 0)))
        dof_maldi = (int(np.sum(np.maximum(np.isfinite(obs_maldi).sum(axis=-2) - 1, 0))) if maldi_n
                     else comp_dof(obs_maldi))
        dof_ngs = (int(np.sum(np.maximum(np.isfinite(obs_ngs).sum(axis=-2) - 1, 0))) if ngs_dm
                   else comp_dof(obs_ngs))
        return [dof_ngs, dof_maldi, int(relative_mask(obs_mibi, 'counts').sum())]
    return [int(np.count_nonzero(np.isfinite(o))) for o in (obs_ngs, obs_maldi, obs_mibi)]


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
    _, obs_maldi = observable_MALDI(days, n_C, s_x, n_C0, const, days)
    _, obs_mibi = observable_MiBi(days, n_C, s_x, n_C0, const, days)
    if _noise_model(calibr_setup) == 'relative':
        n_maldi = maldi_n_of(calibr_setup)
        dm_ngs = ngs_dm_of(calibr_setup)
        sq = [None if dm_ngs else relative_residuals(obs_ngs_meas[i], obs_ngs, 'composition')**2,
              None if n_maldi else relative_residuals(obs_maldi_meas[i], obs_maldi, 'composition')**2,
              relative_residuals(obs_mibi_meas[i], obs_mibi, 'counts')**2]
        sq = [None if q is None else np.where(np.isnan(q), 0., q) for q in sq]  # unused entries: 0 (dropped later)
        if n_maldi:  # failed solve (NaN model) -> inf -> PENALTY in the aggregation
            q = multinomial_residuals(obs_maldi_meas[i], obs_maldi, n_maldi)**2
            sq[1] = np.where(np.isfinite(obs_maldi_meas[i]), np.nan_to_num(q, nan=np.inf, posinf=np.inf), 0.)
        if dm_ngs:
            q = dm_residuals(obs_ngs_meas[i], obs_ngs, *dm_ngs)**2
            sq[0] = np.where(np.isfinite(obs_ngs_meas[i]), np.nan_to_num(q, nan=np.inf, posinf=np.inf), 0.)
        return tuple(sq)
    ll_ngs0 = (obs_ngs_meas[i] - obs_ngs)**2
    ll_maldi0 = (obs_maldi_meas[i] - obs_maldi)**2
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
    With calibr_setup['s_max_penalty'], a 4th group [max(S) - 1] is appended (owner -1);
    total_cost_from_residuals combines the groups exactly like cost_withS.
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
    relative = _noise_model(calibr_setup) == 'relative'
    n_maldi = maldi_n_of(calibr_setup)
    dm_ngs = ngs_dm_of(calibr_setup)
    for obs, mdl_obs, scale, kind in ((obs_ngs_meas, m_ngs, 1., 'composition'),
                                      (obs_maldi_meas, m_maldi, 1., 'composition'),
                                      (obs_mibi_meas, m_mibi, mibi_max, 'counts')):
        obs = np.asarray(obs, dtype=float)
        if n_maldi and mdl_obs is m_maldi:
            mask = np.isfinite(obs)
            r = multinomial_residuals(obs, mdl_obs, n_maldi)
        elif dm_ngs and mdl_obs is m_ngs:
            mask = np.isfinite(obs)
            r = dm_residuals(obs, mdl_obs, *dm_ngs)
        elif relative:
            mask = relative_mask(obs, kind)
            r = relative_residuals(obs, mdl_obs, kind)
        else:
            mask = ~np.isnan(obs)
            r = (obs - mdl_obs) / scale
        res.append(r[mask])
        owner.append(np.broadcast_to(np.arange(len(obs)).reshape((-1,) + (1,) * (obs.ndim - 1)),
                                     obs.shape)[mask])
    if calibr_setup.get('s_max_penalty'):
        res.append(np.array([np.max(s_x) - 1.]))
        owner.append(np.array([-1]))
    return res, owner


def residuals_direct(param, calibr_setup, jac_spasity=None):
    """Signed residuals of cost_direct (S fixed at calibr_setup['s_x'], param = [x0 | ODE]), see residuals_withS."""
    return residuals_withS(np.concatenate([np.asarray(param, dtype=float), np.ravel(calibr_setup['s_x'])]),
                           calibr_setup, jac_spasity)


class TotalCostFromResiduals:
    """
    cost_withS from the residual groups of residuals_withS: aggregation_func of the squared data
    residuals, plus the s_max penalty from the last group if calibr_setup['s_max_penalty'] is set.
    Used by the least-squares fit (reweighting needs the cost as a function of the sums of squares).
    """
    def __init__(self, calibr_setup):
        self.calibr_setup = calibr_setup
        self.agg = calibr_setup['aggregation_func']
        self.penalty = bool(calibr_setup.get('s_max_penalty'))

    def __call__(self, sq_groups):
        if not self.penalty:
            return self.agg(sq_groups)
        return apply_s_max_penalty(self.agg(sq_groups[:-1]), self.calibr_setup, float(np.sum(sq_groups[-1])))


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


def _n_total_points(aggregation_func):
    """Total number of data points of a -2 log L aggregation (GaussianNeg2LogLik and subclasses,
    also inside SelectedDataTypes); None for plain costs such as cost_sum_and_geometric_mean."""
    agg = aggregation_func
    use = None
    if hasattr(agg, 'aggregation_func') and hasattr(agg, 'use'):  # SelectedDataTypes
        agg, use = agg.aggregation_func, agg.use
    n_data = getattr(agg, 'n_data', None)
    if n_data is None:
        return None
    if use is None:
        use = [True] * len(n_data)
    return float(sum(n for n, u in zip(n_data, use) if u))


def s_max_penalty_sq(s_x):
    """(max(S) - 1)^2: deviation of the largest media coefficient from 1."""
    return float((np.max(s_x) - 1.)**2)


def apply_s_max_penalty(cost, calibr_setup, p_sq):
    """
    Add the penalty for max(S) != 1 to the aggregated cost, with weight w =
    calibr_setup['s_max_penalty'] (None/0: no penalty). The penalty acts on the log scale, so that
    calibration cost and profile objectives have the same optimum:
      plain (positive) costs, e.g. cost_sum_and_geometric_mean:  cost * exp(w * (max S - 1)^2)
      -2 log L objectives (GaussianNeg2LogLik, GeometricMeanNeg2LogLik): cost + N * w * (max S - 1)^2
    (N = number of data points). Interpretation: max(S) = 1 - d raises the cost by the factor
    exp(w d^2), e.g. w = 100, d = 0.1: +172 %; d = 0.03: +9.4 %.
    """
    w = calibr_setup.get('s_max_penalty')
    if not w:
        return cost
    n_total = _n_total_points(calibr_setup['aggregation_func'])
    if n_total is not None:
        return cost + n_total * float(w) * p_sq
    if cost <= 0:
        raise ValueError('s_max_penalty needs a positive cost or a -2 log L aggregation')
    return cost * np.exp(float(w) * p_sq)


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
    #ll_s1, ll_s2 = S_squared_difference(param[-n_cl*n_media:], calibr_setup['dfs'], calibr_setup['data_array'], calibr_setup['T_x'])
    cost = calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_mibi])#, ll_s1, ll_s2])
    # optional penalty so that the largest media coefficient is 1 (calibr_setup['s_max_penalty'] = weight)
    return apply_s_max_penalty(cost, calibr_setup, s_max_penalty_sq(s_x))


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
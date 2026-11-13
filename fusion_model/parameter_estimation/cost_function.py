from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, polygamma
from scipy.stats.mstats import gmean
from fusion_model.model.solving import model_ODE_solution, get_bacterial_count
from fusion_model.model.solving import observable_NGS, observable_MALDI, observable_platecount
from fusion_model.parameter_estimation.media_matrix import S_squared_difference


# ----------------------------------------------------------------------
# Noise model of the residuals: calibr_setup['noise_model']
#   'absolute' (default): residual = measured - model (plate counts divided by their maximum)
#   'relative': for multiplicative noise, measured = model * (1 + eps):
#       plate counts (counts):          log(measured) - log(model)
#       MALDI, NGS (fractions): centred log-ratio residual within each composition (over the species of one
#                               experiment / medium / time point): log-ratios are not affected by the
#                               normalization to proportions, the noise becomes eps_i - mean(eps)
#   Entries with measured <= 0 (below detection, absent species) are not used; a composition with fewer than
#   2 present species carries no information. Each composition has (present species - 1) degrees of freedom
#   (count_data_points_noise), which -2 log L objectives must use as the number of data points.
# ----------------------------------------------------------------------
#
# calibr_setup['maldi_n'] = N, one number or an array of N per composition (shape of the MALDI data with a species axis
#   of length 1, e.g. the colonies identified on each plate; None: off, needs noise_model 'relative' and pl.make_gaussian_neg2loglik(..., maldi_n=N)):
#   MALDI as N identified colonies per plate, counts k_j = N * measured ~ Multinomial(N, p_j(model)). Residuals
#   are the deviance residuals of the counts (multinomial_deviance_residuals), one per species incl. the undetected
#   ones (k_j = 0); their sum of squares is -2 log L of the multinomial up to a constant, with no noise level to estimate.
# calibr_setup['maldi_dm'] = (N, A): MALDI as Dirichlet-multinomial instead of the multinomial maldi_n, N as for maldi_n.
# calibr_setup['ngs_dm'] = (N, A), N one number or an array of the reads of each sample as for maldi_n (None: off, needs noise_model 'relative' and pl.make_gaussian_neg2loglik(..., ngs_dm=(N, A))):
#   NGS as N reads per sample with species counts k ~ Dirichlet-multinomial(N, A p), including k_j = 0. The residuals
#   are dm_deviance_residuals; their sum of squares is -2 log L up to a constant, with N and A known.
# ----------------------------------------------------------------------
NOISE_MODELS = ('absolute', 'relative')


def _noise_model(calibr_setup):
    nm = calibr_setup.get('noise_model') or 'absolute'
    if nm not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {nm!r}")
    return nm


def count_setting_on(x):
    """True if a count setting (calibr_setup['maldi_n'] or ['ngs_dm']) is given: a number, an array of numbers per
    composition, or a tuple (N, A) with N one of these."""
    if x is None:
        return False
    if isinstance(x, (tuple, list)) and len(x) == 2 and np.ndim(x[1]) == 0:  # (N, A)
        return count_setting_on(x[0])
    return not (np.ndim(x) == 0 and not x)


def _counts_per_composition(n):
    """N as given: int for one number (as before), else a float array of N per composition (species axis of length 1,
    broadcast against the data); missing compositions (NaN) get 1, they are masked out anyway."""
    if np.ndim(n) == 0:
        return int(n)
    n = np.asarray(n, dtype=float)
    return np.where(np.isfinite(n) & (n > 0), n, 1.)


def maldi_n_of(calibr_setup):
    """Number of identified colonies per plate if MALDI is fitted with the multinomial likelihood, else None;
    checks that the noise model and the objective fit to it."""
    n = calibr_setup.get('maldi_n')
    agg = calibr_setup.get('aggregation_func')
    agg = getattr(agg, 'aggregation_func', agg)  # inside SelectedDataTypes
    is_multinomial = bool(getattr(agg, 'is_deviance', (False, False, False))[1])
    if not count_setting_on(n):
        if is_multinomial and not count_setting_on(calibr_setup.get('maldi_dm')):
            raise ValueError("aggregation MultinomialMaldiNeg2LogLik needs calibr_setup['maldi_n'] or ['maldi_dm']")
        return None
    if count_setting_on(calibr_setup.get('maldi_dm')):
        raise ValueError("calibr_setup: give either 'maldi_n' (multinomial) or 'maldi_dm' (Dirichlet-multinomial)")
    if _noise_model(calibr_setup) != 'relative':
        raise ValueError("calibr_setup['maldi_n'] needs noise_model = 'relative'")
    if not is_multinomial:
        raise ValueError("calibr_setup['maldi_n'] needs the objective pl.make_gaussian_neg2loglik(..., maldi_n=N) "
                         f"('neg2logL'), got {type(agg).__name__ if agg is not None else None}")
    return _counts_per_composition(n)


def maldi_dm_of(calibr_setup):
    """(colonies per plate N, concentration A) if MALDI is fitted with the Dirichlet-multinomial likelihood
    (calibr_setup['maldi_dm'], N one number or one per composition as for maldi_n), else None."""
    dm = calibr_setup.get('maldi_dm')
    if not count_setting_on(dm):
        return None
    agg = calibr_setup.get('aggregation_func')
    agg = getattr(agg, 'aggregation_func', agg)  # inside SelectedDataTypes
    if _noise_model(calibr_setup) != 'relative':
        raise ValueError("calibr_setup['maldi_dm'] needs noise_model = 'relative'")
    if not bool(getattr(agg, 'is_deviance', (False, False, False))[1]):
        raise ValueError("calibr_setup['maldi_dm'] needs the objective pl.make_gaussian_neg2loglik(..., maldi_dm=(N, A))")
    return _counts_per_composition(dm[0]), float(dm[1])


def ngs_dm_of(calibr_setup):
    """(reads per sample N, concentration A) if NGS is fitted with the Dirichlet-multinomial likelihood, else None;
    checks that the noise model and the objective fit to it."""
    dm = calibr_setup.get('ngs_dm')
    agg = calibr_setup.get('aggregation_func')
    agg = getattr(agg, 'aggregation_func', agg)  # inside SelectedDataTypes
    is_dm = bool(getattr(agg, 'is_deviance', (False, False, False))[0])
    if not count_setting_on(dm):
        if is_dm:
            raise ValueError("this aggregation has NGS as deviance and needs calibr_setup['ngs_dm']")
        return None
    if _noise_model(calibr_setup) != 'relative':
        raise ValueError("calibr_setup['ngs_dm'] needs noise_model = 'relative'")
    if not is_dm:
        raise ValueError("calibr_setup['ngs_dm'] needs the objective pl.make_gaussian_neg2loglik(..., ngs_dm=(N, A)) "
                         f"('neg2logL'), got {type(agg).__name__ if agg is not None else None}")
    return _counts_per_composition(dm[0]), float(dm[1])


def multinomial_counts(meas, n):
    """Counts n * meas of proportions (NaN stays NaN); error if they are not whole numbers, i.e. if the data do
    not come from n identified colonies per plate (MALDI) or n reads per sample (NGS)."""
    k = np.asarray(meas, dtype=float) * n
    if np.nanmax(np.abs(k - np.round(k)), initial=0.) > 1e-6:
        what = n if np.ndim(n) == 0 else 'N (one per composition)'
        raise ValueError(f'proportions are not multiples of 1/{what}: the data do not have these counts per composition')
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
    """Lagrange multiplier c = log(1 + N / A) of the Dirichlet-multinomial residuals (dm_deviance_residuals);
    an array for N per composition."""
    c = np.log1p(np.asarray(n, dtype=float) / conc)
    return float(c) if np.ndim(c) == 0 else c


def dm_saturated(k, n, conc):
    """
    Per-entry maximum of g_k(a) = log Gamma(k + a) - log Gamma(a) - c a, a = A p, c = _dm_rate(n, conc), for the counts k.
    k = 0: a* = 0, g* = 0. k > 0: g_k is concave, a* solves digamma(k + a) - digamma(a) = c.
    Returns arrays of the shape of k: a*, g*, s2 = sqrt(-g''(a*)) and s3 = g'''(a*) / (6 g''(a*)) (series near a*).
    """
    k = np.asarray(k, dtype=float)
    c_all = np.broadcast_to(np.asarray(_dm_rate(n, conc), dtype=float), k.shape)
    pairs, inverse = np.unique(np.stack([k.ravel(), c_all.ravel()], axis=1), axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    values = pairs[:, 0]
    a_star, g_star, s2, s3 = np.zeros(values.shape), np.zeros(values.shape), np.ones(values.shape), np.zeros(values.shape)
    for i, (v, c) in enumerate(pairs):
        if v <= 0:
            continue
        lo = v / np.expm1(c)  # sum_i 1 / (a + i) >= log(1 + k / a), so a* >= k / (exp(c) - 1)
        lo = lo * (1. - 1e-9)
        hi = 2. * lo + 1.
        root = lambda a: (polygamma(0, v + a) - polygamma(0, a)) - c
        if not root(lo) > 0:  # very large a (A >> N): digamma differences lose precision, use the expansion
            a = v / c - (v - 1.) / 2.
        else:
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
    """dm_saturated for the numpy cost: the counts are fixed during a fit, so it is computed once per dataset.
    n: a number, or (bytes, shape) of an array of N per composition."""
    if isinstance(n, tuple):
        n = np.frombuffer(n[0]).reshape(n[1])
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
    n_key = n if np.ndim(n) == 0 else (np.ascontiguousarray(n, dtype=float).tobytes(), np.shape(n))
    sat = _dm_saturated_cached(k0.tobytes(), k0.shape, n_key, conc)
    r = dm_deviance_residuals(k0, np.where(use, np.asarray(model, dtype=float), 1.) * conc, sat, _dm_rate(n, conc))
    return np.where(use, r, np.nan)


def dm_neg2loglik(k, p, n, conc):
    """
    -2 log L of Dirichlet-multinomial counts k (species on axis -2, NaN: not measured) with proportions p, totals n
    (one number or one per composition, species axis of length 1) and concentration A = conc, without the term
    log(N! / prod k_j!) that depends on the data only. Unlike the deviance residuals it contains every term that depends
    on A, so it can be minimized over A: dm_concentration_mle. Categories with k = 0 and p = 0 (species fixed absent)
    contribute 0.
    """
    k = np.asarray(k, dtype=float)
    use = np.isfinite(k)
    a = conc * np.where(use, np.asarray(p, dtype=float), 0.)
    k0 = np.where(use, k, 0.)
    absent = (k0 == 0) & (a <= 0)
    a_safe = np.where(absent, 1., np.maximum(a, 1e-300))
    per_species = np.where(absent | ~use, 0., gammaln(k0 + a_safe) - gammaln(a_safe))
    measured = use.any(axis=-2)
    n_comp = np.broadcast_to(np.asarray(n, dtype=float), k.shape[:-2] + (1,) + k.shape[-1:])[..., 0, :]
    per_comp = gammaln(conc) - gammaln(np.where(measured, n_comp, 0.) + conc)
    return float(-2. * (np.sum(per_species) + np.sum(np.where(measured, per_comp, 0.))))


def dm_concentration_mle(k, p, n, bounds=(1e-1, 1e7)):
    """Maximum-likelihood concentration A of Dirichlet-multinomial counts for given proportions p (see dm_neg2loglik),
    on a log scale within bounds. Returns (A, -2 log L at A)."""
    from scipy.optimize import minimize_scalar
    f = lambda la: dm_neg2loglik(k, p, n, 10.**la)
    r = minimize_scalar(f, bounds=np.log10(bounds), method='bounded', options={'xatol': 1e-4})
    return float(10.**r.x), float(r.fun)


def relative_mask(meas, kind):
    """Data entries used by the relative noise model (depends only on the data).
    kind: 'counts' (plate counts) or 'composition' (MALDI, NGS; species on axis -2)."""
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


def count_data_points_noise(data_array, noise_model='absolute', maldi_n=None, ngs_dm=None, maldi_dm=None):
    """Number of data points (degrees of freedom) per data type [ngs, maldi, platecount] for a noise model.
    maldi_n: MALDI fitted with the multinomial likelihood, (species - 1) per measured composition incl. zeros.
    ngs_dm: the same for NGS fitted with the Dirichlet-multinomial likelihood."""
    _, (obs_platecount, obs_maldi, obs_ngs) = data_array
    if noise_model == 'relative':
        def comp_dof(obs):
            use = relative_mask(obs, 'composition')
            return int(np.sum(np.maximum(use.sum(axis=-2) - 1, 0)))
        dof_maldi = (int(np.sum(np.maximum(np.isfinite(obs_maldi).sum(axis=-2) - 1, 0))) if (count_setting_on(maldi_n) or count_setting_on(maldi_dm))
                     else comp_dof(obs_maldi))
        dof_ngs = (int(np.sum(np.maximum(np.isfinite(obs_ngs).sum(axis=-2) - 1, 0))) if count_setting_on(ngs_dm)
                   else comp_dof(obs_ngs))
        return [dof_ngs, dof_maldi, int(relative_mask(obs_platecount, 'counts').sum())]
    return [int(np.count_nonzero(np.isfinite(o))) for o in (obs_ngs, obs_maldi, obs_platecount)]


def ode_params_for_model(param_ode, calibr_setup, xp=np):
    """
    ODE parameters in the form the model functions expect. Default: unchanged.
    calibr_setup['lambda_tref'] = T_ref (one value or one per species): the first n_cl entries of param_ode are
    lambda_ref = log10 lambda(T_ref) instead of lambda_1, i.e. lambda(T) = 10^lambda_ref * (T / T_ref)^lambda_exp.
    Same model, it only decorrelates the two lambda parameters: lambda_1 = lambda_ref - lambda_exp * log10(T_ref).
    """
    t_ref = calibr_setup.get('lambda_tref')
    structure = calibr_setup.get('k_structure') or 'full'
    if t_ref is None and structure == 'full':
        return param_ode
    n_cl = calibr_setup['n_cl']
    ode = xp.asarray(param_ode)
    if t_ref is not None:
        ode = xp.concatenate([ode[:n_cl] - ode[n_cl:2*n_cl] * np.log10(t_ref), ode[n_cl:]])
    if structure != 'full':
        ode = xp.concatenate([ode[:4*n_cl + 2], xp.ravel(structured_k(ode[4*n_cl + 2:], n_cl, structure, xp=xp,
                                                                      groups=calibr_setup.get('k_groups')))])
    return ode


# ----------------------------------------------------------------------
# Reduced inhibition matrices: calibr_setup['k_structure'] (fusion_model2 layout, k_ij = last n_cl x n_cl entries of the
# ODE parameters; k_ij = effect of species j on the growth of species i, diagonal 0)
#   'full'    (default) every k_ij is a parameter: n (n - 1)
#   'column'  k_ij = b_j: species j inhibits all others equally: n parameters
#   'product' k_ij = a_i b_j: sensitivity of species i times strength of species j, a_0 = 1: 2 n - 1 parameters
#   'shared'  k_ij = a_i b_j also for i = j: every species feels the same inhibitor load I = sum_j b_j x_j (its own
#             products included) with its sensitivity a_i; a_0 = 1: 2 n - 1 parameters, from n = 2
#   'shared_c' as 'shared' with the diagonal scaled by a free factor c: k_ii = c a_i b_i (c = 0: 'product', c = 1:
#             'shared'), c in k[0, 1]; 2 n parameters, from n = 3
#   'none'    k_ij = 0
# The parameter vector keeps its layout. The reduced parameters live in entries of the k block that are otherwise unused:
# b_j in k[(j + 1) mod n, j] and a_i in k[i, (i + 1) mod n] (n >= 3); for 'shared' a_i in k[i, i] and b_j in
# k[(j + 1) mod n, j] (n >= 2). All other entries are fixed at 0 and ignored.
# k_structure_bounds gives the bounds for a structure, k_to_structure the start point from a full matrix.
# calibr_setup['k_groups'] (with 'product', 'shared', 'shared_c'): one group label per species; species of one group share one strength b,
# stored in the b entry of the group's first species (the other b entries are fixed at 0 and ignored). The sensitivities
# a_i stay per species. Groups can come from taxonomy or from the data (fusion_model.analysis.inhibition_groups).
# ----------------------------------------------------------------------
K_STRUCTURES = ('full', 'column', 'product', 'shared', 'shared_c', 'none')


C_SLOT = (0, 1)  # entry of the self-inhibition factor c of 'shared_c'


def _k_slots(n_cl, structure='product'):
    idx = np.arange(n_cl)
    if structure in ('shared', 'shared_c'):
        return (idx, idx), ((idx + 1) % n_cl, idx)  # a_i on the diagonal, b_j below it
    return (idx, (idx + 1) % n_cl), ((idx + 1) % n_cl, idx)  # entries of a_i, entries of b_j


def _group_representatives(groups):
    """Index of the representative (first species) of each species' group: [rep(g_0), rep(g_1), ...]."""
    first = {}
    for j, g in enumerate(groups):
        first.setdefault(g, j)
    return np.array([first[g] for g in groups])


def structured_k(k_block, n_cl, structure, xp=np, groups=None):
    """Full n_cl x n_cl inhibition matrix from the k block of a parameter vector with a reduced structure;
    groups: species groups sharing one strength b (only 'shared', see calibr_setup['k_groups'])."""
    if structure not in K_STRUCTURES:
        raise ValueError(f'k_structure must be one of {K_STRUCTURES}, got {structure!r}')
    k = xp.reshape(xp.asarray(k_block), (n_cl, n_cl))
    if structure == 'full':
        return k
    if structure == 'none':
        return k * 0.
    if structure in ('shared', 'shared_c'):
        if n_cl < (3 if structure == 'shared_c' else 2):
            raise ValueError(f"k_structure {structure!r} needs at least {3 if structure == 'shared_c' else 2} species")
        slot_a, slot_b = _k_slots(n_cl, 'shared')
        b = k[slot_b]
        if groups is not None:
            b = b[_group_representatives(groups)]
        full = k[slot_a][:, None] * b[None, :]
        if structure == 'shared_c':
            eye = np.eye(n_cl)
            full = full * (1. - eye) + full * eye * k[C_SLOT]
        return full
    if n_cl < 3:
        raise ValueError("k_structure 'column' / 'product' needs at least 3 species")
    slot_a, slot_b = _k_slots(n_cl)
    b = k[slot_b]
    if groups is not None:
        if structure != 'product':
            raise ValueError("k_groups needs k_structure 'product', 'shared' or 'shared_c'")
        b = b[_group_representatives(groups)]
    full = xp.ones((n_cl, 1)) * b[None, :]
    if structure == 'product':
        full = k[slot_a][:, None] * full
    return full * (1. - np.eye(n_cl))


def k_block_start(param_bnds, n_cl):
    """Index of the first k entry in a parameter vector [x0 | ODE (4 n + 2 + n^2) | S (n_media x n)] or [x0 | ODE]."""
    n = len(param_bnds)
    for n_s in range(n // n_cl, -1, -1):  # with S: prefer the largest S block that leaves whole experiments of x0
        n_x0 = n - n_s * n_cl - (4 * n_cl + 2 + n_cl * n_cl)
        if n_x0 >= 0 and n_x0 % n_cl == 0:
            return n_x0 + 4 * n_cl + 2
    raise ValueError('parameter vector does not have the fusion_model2 layout')


def k_structure_bounds(param_bnds, n_cl, structure, k_start, k_bounds=(0., 20.), a_bounds=(0.01, 100.), groups=None,
                       c_bounds=(0., 10.)):
    """param_bnds with the k block (from index k_start) set for a structure: b_j in k_bounds, a_i in a_bounds with
    a_0 = 1 (the common scale of a and b is free otherwise), unused entries fixed at 0. 'full': unchanged."""
    if structure == 'full':
        return tuple(tuple(b) for b in param_bnds)
    bnds = [tuple(b) for b in param_bnds]
    block = np.zeros((n_cl, n_cl, 2))
    slot_a, slot_b = _k_slots(n_cl, structure)
    if structure in ('column', 'product', 'shared', 'shared_c'):
        block[slot_b] = k_bounds
    if structure in ('product', 'shared', 'shared_c'):
        block[slot_a] = a_bounds
        block[slot_a[0][0], slot_a[1][0]] = (1., 1.)  # a_0 = 1
    if structure == 'shared_c':
        block[C_SLOT] = c_bounds
    if groups is not None:
        if structure not in ('product', 'shared', 'shared_c'):
            raise ValueError("k_groups needs k_structure 'product', 'shared' or 'shared_c'")
        rep = _group_representatives(groups)
        for j in range(n_cl):
            if rep[j] != j:
                block[slot_b[0][j], slot_b[1][j]] = (0., 0.)  # strength of the group's first species only
    bnds[k_start:k_start + n_cl * n_cl] = [tuple(v) for v in block.reshape(-1, 2)]
    return tuple(bnds)


def k_to_structure(param, n_cl, structure, k_start, a_bounds=(0.01, 100.), groups=None):
    """Parameter vector with the full k block replaced by its closest reduced form, as a start point: 'column':
    b_j = mean of column j; 'product': leading singular vectors of k (non-negative), a_0 = 1; 'none': 0."""
    p = np.array(param, dtype=float)
    if structure == 'full':
        return p
    k = p[k_start:k_start + n_cl * n_cl].reshape(n_cl, n_cl)
    off = ~np.eye(n_cl, dtype=bool)
    new = np.zeros((n_cl, n_cl))
    slot_a, slot_b = _k_slots(n_cl, structure)
    if structure == 'column':
        new[slot_b] = [k[off[:, j], j].mean() for j in range(n_cl)]
    elif structure in ('product', 'shared', 'shared_c'):
        u, s, vt = np.linalg.svd(k)
        a, b = np.abs(u[:, 0]) * np.sqrt(s[0]), np.abs(vt[0]) * np.sqrt(s[0])
        a = np.clip(a / max(a[0], 1e-12), *a_bounds)
        a[0] = 1.
        b = np.array([np.sum(a[off[:, j]] * k[off[:, j], j]) / np.sum(a[off[:, j]]**2) for j in range(n_cl)])  # least squares given a
        new[slot_a], new[slot_b] = a, np.maximum(b, 0.)
    if structure == 'shared_c':
        new[C_SLOT] = 1.
    if groups is not None and structure in ('product', 'shared', 'shared_c'):  # group strength: mean of the members' strengths
        rep = _group_representatives(groups)
        b = new[slot_b].copy()
        for r in set(rep):
            new[slot_b[0][r], slot_b[1][r]] = b[rep == r].mean()
        for j in range(n_cl):
            if rep[j] != j:
                new[slot_b[0][j], slot_b[1][j]] = 0.
    p[k_start:k_start + n_cl * n_cl] = new.ravel()
    return p


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, s_x, platecount_max):
    """Simulate one experiment and return squared residuals for NGS, MALDI, and plate counts."""
    #temp = float(df_platecount0.columns[0].split("_")[2][:-1])
    model = calibr_setup['model']
    days, [obs_platecount_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    temp = calibr_setup['exp_temps'][exp]
    T_x = calibr_setup['T_x']
    const = [[temp], n_cl, calibr_setup['media']]

    C0 = np.concatenate((10**np.array(x0), np.ones((n_cl+1))))
    C = model_ODE_solution(model, days, param_ode, C0, const)
    n_C = get_bacterial_count(C, np.shape(s_x)[-1], 2)
    n_C0 = get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(s_x)[-1], 2)

    _, obs_ngs = observable_NGS(days, n_C, T_x, n_C0, const, days)
    _, obs_maldi = observable_MALDI(days, n_C, s_x, n_C0, const, days)
    _, obs_platecount = observable_platecount(days, n_C, s_x, n_C0, const, days)
    if _noise_model(calibr_setup) == 'relative':
        n_maldi = maldi_n_of(calibr_setup)
        dm_ngs = ngs_dm_of(calibr_setup)
        # counts per composition: this experiment's entries
        if n_maldi is not None and np.ndim(n_maldi) > 0:
            n_maldi = n_maldi[i]
        if dm_ngs is not None and np.ndim(dm_ngs[0]) > 0:
            dm_ngs = (dm_ngs[0][i], dm_ngs[1])
        dm_maldi = maldi_dm_of(calibr_setup)
        if dm_maldi is not None and np.ndim(dm_maldi[0]) > 0:
            dm_maldi = (dm_maldi[0][i], dm_maldi[1])
        sq = [None if dm_ngs is not None else relative_residuals(obs_ngs_meas[i], obs_ngs, 'composition')**2,
              None if (n_maldi is not None or dm_maldi is not None)
              else relative_residuals(obs_maldi_meas[i], obs_maldi, 'composition')**2,
              relative_residuals(obs_platecount_meas[i], obs_platecount, 'counts')**2]
        sq = [None if q is None else np.where(np.isnan(q), 0., q) for q in sq]  # unused entries: 0 (dropped later)
        if n_maldi is not None:  # failed solve (NaN model) -> inf -> PENALTY in the aggregation
            q = multinomial_residuals(obs_maldi_meas[i], obs_maldi, n_maldi)**2
            sq[1] = np.where(np.isfinite(obs_maldi_meas[i]), np.nan_to_num(q, nan=np.inf, posinf=np.inf), 0.)
        if dm_maldi is not None:
            q = dm_residuals(obs_maldi_meas[i], obs_maldi, *dm_maldi)**2
            sq[1] = np.where(np.isfinite(obs_maldi_meas[i]), np.nan_to_num(q, nan=np.inf, posinf=np.inf), 0.)
        if dm_ngs is not None:
            q = dm_residuals(obs_ngs_meas[i], obs_ngs, *dm_ngs)**2
            sq[0] = np.where(np.isfinite(obs_ngs_meas[i]), np.nan_to_num(q, nan=np.inf, posinf=np.inf), 0.)
        return tuple(sq)
    ll_ngs0 = (obs_ngs_meas[i] - obs_ngs)**2
    ll_maldi0 = (obs_maldi_meas[i] - obs_maldi)**2
    ll_platecount0 = (obs_platecount_meas[i] - obs_platecount)**2 /platecount_max
    return ll_ngs0, ll_maldi0, ll_platecount0


def squared_differences(param_ode, x0_vals, s_x, calibr_setup, jac_spasity=None):
    """Collect the nonzero observation residuals across all calibration experiments."""
    param_ode = ode_params_for_model(param_ode, calibr_setup)
    (df_platecount, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, [obs_platecount_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    n_cl = np.shape(obs_maldi_meas)[2] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_platecount.columns])))
    platecount_max = np.nanmax(obs_platecount_meas, axis=(0, 1))**2
    ll_ngs, ll_maldi, ll_platecount = np.zeros(np.shape(obs_ngs_meas)), np.zeros(np.shape(obs_maldi_meas)), np.zeros(np.shape(obs_platecount_meas))
    for i, exp in enumerate(exps):
        ll_ngs[i], ll_maldi[i], ll_platecount[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, s_x, platecount_max)
    return ll_ngs[ll_ngs!=0], ll_maldi[ll_maldi!=0], ll_platecount[ll_platecount!=0]


def residuals_withS(param, calibr_setup, jac_spasity=None):
    """
    Signed residuals of cost_withS, for least-squares solvers:
    returns ([r_ngs, r_maldi, r_platecount], [exp_ngs, exp_maldi, exp_platecount]) with
    r_k = measured - model (plate counts divided by their maximum, as in sq_diff_oneexp) for all
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

    (df_platecount, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, meas = calibr_setup['data_array']
    obs_platecount_meas, obs_maldi_meas, obs_ngs_meas = meas
    n_cl_d = np.shape(obs_maldi_meas)[2]
    exps = sorted(list(set([s.split('_')[0] for s in df_platecount.columns])))
    platecount_max = np.nanmax(obs_platecount_meas, axis=(0, 1))
    param_ode = ode_params_for_model(param_ode, calibr_setup)
    model = calibr_setup['model']
    T_x = calibr_setup['T_x']
    m_ngs, m_maldi, m_platecount = (np.full(np.shape(obs_ngs_meas), np.nan), np.full(np.shape(obs_maldi_meas), np.nan),
                              np.full(np.shape(obs_platecount_meas), np.nan))
    for i, exp in enumerate(exps):
        const = [[calibr_setup['exp_temps'][exp]], n_cl_d, calibr_setup['media']]
        C0 = np.concatenate((10**np.array(x0_vals[n_cl_d*i:n_cl_d*(i+1)]), np.ones((n_cl_d+1))))
        C = model_ODE_solution(model, days, param_ode, C0, const)
        n_C = get_bacterial_count(C, np.shape(s_x)[-1], 2)
        n_C0 = get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(s_x)[-1], 2)
        m_ngs[i] = observable_NGS(days, n_C, T_x, n_C0, const, days)[1]
        m_maldi[i] = observable_MALDI(days, n_C, s_x, n_C0, const, days)[1]
        m_platecount[i] = observable_platecount(days, n_C, s_x, n_C0, const, days)[1]

    res, owner = [], []
    relative = _noise_model(calibr_setup) == 'relative'
    n_maldi = maldi_n_of(calibr_setup)
    dm_ngs = ngs_dm_of(calibr_setup)
    dm_maldi = maldi_dm_of(calibr_setup)
    for obs, mdl_obs, scale, kind in ((obs_ngs_meas, m_ngs, 1., 'composition'),
                                      (obs_maldi_meas, m_maldi, 1., 'composition'),
                                      (obs_platecount_meas, m_platecount, platecount_max, 'counts')):
        obs = np.asarray(obs, dtype=float)
        if n_maldi is not None and mdl_obs is m_maldi:
            mask = np.isfinite(obs)
            r = multinomial_residuals(obs, mdl_obs, n_maldi)
        elif dm_maldi is not None and mdl_obs is m_maldi:
            mask = np.isfinite(obs)
            r = dm_residuals(obs, mdl_obs, *dm_maldi)
        elif dm_ngs is not None and mdl_obs is m_ngs:
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


def cost_withS_withdiffmu(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost with experiment-specific mu parameters and fitted media weights."""
    n_cl = calibr_setup['n_cl']
    n_media = calibr_setup['n_media']
    exps = calibr_setup['exps']
    lambd  = param[n_cl*len(exps):n_cl*len(exps)+n_cl]
    mu = param[n_cl*len(exps)+n_cl:n_cl*len(exps)+n_cl + n_cl*len(exps)]
    rest_ode_param = param[n_cl*len(exps)+n_cl + n_cl*len(exps):-n_cl*n_media]
    s_x = np.array(param)[-n_cl*n_media:].reshape((n_media, n_cl))
    x0_vals = param[:n_cl*len(exps)]

    (df_platecount, df_maldi, df_ngs, ) = calibr_setup['dfs']
    days, [obs_platecount_meas, obs_maldi_meas, obs_ngs_meas] = calibr_setup['data_array']
    n_cl = np.shape(obs_maldi_meas)[2] #np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split('_')[0] for s in df_platecount.columns])))
    platecount_max = np.nanmax(obs_platecount_meas, axis=(0, 1))**2
    ll_ngs, ll_maldi, ll_platecount = np.zeros(np.shape(obs_ngs_meas)), np.zeros(np.shape(obs_maldi_meas)), np.zeros(np.shape(obs_platecount_meas))
    for i, exp in enumerate(exps):
        param_ode = np.concatenate((lambd, mu[n_cl*i:n_cl*(i+1)], rest_ode_param))
        ll_ngs[i], ll_maldi[i], ll_platecount[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, s_x, platecount_max)
    ll_ngs = ll_ngs[ll_ngs!=0]
    ll_maldi = ll_maldi[ll_maldi!=0]
    ll_platecount = ll_platecount[ll_platecount!=0]
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_platecount])


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
    ll_ngs, ll_maldi, ll_platecount = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    #ll_s1, ll_s2 = S_squared_difference(param[-n_cl*n_media:], calibr_setup['dfs'], calibr_setup['data_array'], calibr_setup['T_x'])
    cost = calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_platecount])#, ll_s1, ll_s2])
    # optional penalty so that the largest media coefficient is 1 (calibr_setup['s_max_penalty'] = weight)
    return apply_s_max_penalty(cost, calibr_setup, s_max_penalty_sq(s_x))


def cost_direct(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost using media weights fixed in the setup."""
    s_x = calibr_setup['s_x']
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = param[n_cl*len(exps):]
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_platecount = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_platecount])


def cost_initvals(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost while fitting experiment-specific initial values only."""
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    ll_ngs, ll_maldi, ll_platecount = squared_differences(param_ode, x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_platecount])


def cost_initvals_lambda(param, calibr_setup, jac_spasity):
    """Evaluate the calibration cost while fitting initial values and lambda parameters."""
    n_cl = calibr_setup['n_cl']
    exps = calibr_setup['exps']
    param_ode = calibr_setup['param_ode']
    s_x = calibr_setup['s_x']
    x0_vals = param[:n_cl*len(exps)]
    lambd = param[n_cl*len(exps):]
    ll_ngs, ll_maldi, ll_platecount = squared_differences(np.concatenate((lambd, param_ode)), x0_vals, s_x, calibr_setup)
    return calibr_setup['aggregation_func']([ll_ngs, ll_maldi, ll_platecount])


def cost_logsumexp(J_vect):#ll_ngs, ll_maldi, ll_platecount):
    """Combine residual arrays with a smooth log-sum-exp aggregate, ignoring NaNs."""
    a = .1
    num_meas = np.sum([np.size(Ji[~np.isnan(Ji)]) for Ji in J_vect])
    return np.log(np.nansum([np.nansum(np.exp(a*Ji)) for Ji in J_vect])/num_meas)/a


def cost_geometric_mean(J_vect):#ll_ngs, ll_maldi, ll_platecount):
    """Compute the geometric mean across the supplied residual values, ignoring NaNs."""
    num_meas = np.sum([np.size(Ji[~np.isnan(Ji)]) for Ji in J_vect])
    return np.nanprod([Ji**(1/num_meas) for Ji in J_vect])


def cost_arithmetic_mean(J_vect):#ll_ngs, ll_maldi, ll_platecount):
    """Return the mean of each residual array's mean, ignoring NaNs."""
    return np.nanmean([np.nanmean(Ji) for Ji in J_vect])


def cost_sum_and_geometric_mean(J_vect):#ll_ngs, ll_maldi, ll_platecount):
    """Return the geometric mean of the non-NaN residual sum from each array."""
    return gmean([np.nansum(Ji) for Ji in J_vect], nan_policy='omit', axis=None)


DATA_TYPES = ('ngs', 'maldi', 'platecount')  # order of J_vect in cost_withS / squared_differences


class SelectedDataTypes:
    """
    Aggregation over a subset of the data types: calls aggregation_func only with the
    residual arrays of the data types in use (names from DATA_TYPES), e.g.
    SelectedDataTypes(cost_sum_and_geometric_mean, ['ngs', 'maldi']) calibrates without plate counts.
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
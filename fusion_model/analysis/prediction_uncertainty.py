"""
Prediction uncertainty from parameter uncertainty
=================================================

Propagates the uncertainty of the estimated parameters to the model predictions
(bacterial counts x(t), plate counts, MALDI and NGS proportions):

    1) parameter ensemble: parameter vectors compatible with the data, i.e. with
       cost - cost_opt <= chi2.ppf(confidence_level, dof) * scale
         * ensemble_from_profile_csv : all points of a profile-likelihood run (run_profile_likelihood_all,
                                       run_profile_likelihood_adaptive, ...) below the threshold; every
                                       point is a full, re-optimized parameter vector
         * ensemble_from_fits        : any set of fits with their costs, e.g. the end points of
                                       multistart_optimization, or parameter samples
    2) predict_ensemble : simulate every ensemble member for the requested experiments
    3) prediction_bands : pointwise band over the members ('envelope' = min/max, or quantiles)
    4) fm.plotting.plot_prediction_uncertainty : band, optionally with the estimate and the data

Example (calibration experiments, their x0 come from the parameter vectors):

    ens = fm.pred_unc.ensemble_from_profile_csv(csv_path, exps=exps_calibr)
    pred = fm.pred_unc.predict_ensemble(ens, calibr_setup, np.linspace(0, 17, 200))
    fm.plotting.plot_prediction_uncertainty(pred, calibr_setup, path=path, add_name='_calibration')

Example (new experiments, only ODE parameters and S are taken from the ensemble):

    pred = fm.pred_unc.predict_ensemble(ens, prediction_setup, t, exps=exps_predict, x0=x0_dict,
                                        refit_x0=True, n_jobs=8)

Initial values of new experiments (x0):
    refit_x0=False : the given x0 (e.g. the prediction estimate) for all members; only the uncertainty
                     of the ODE parameters and S is propagated.
    refit_x0=True  : x0 is re-estimated for each member on the data in setup['dfs'] (cost_initvals,
                     L-BFGS-B started at x0), so each member has its own best x0, as the estimate has.

Parameter vector layout (as in cost_withS / cost_direct):
    [ x0 (n_cl per calibration experiment, log10) | ODE parameters | S (n_media x n_cl, only if with_S) ]
Without S in the vector (e.g. cost_direct / separately estimated S) setup['s_x'] is used for all members.

Note on the profile-based ensemble: the profiles are paths through the confidence region, not a sample
of it. The envelope of their predictions is the usual, cheap approximation of the prediction confidence
band (lower bound of the true band); a prediction profile likelihood (Kreutz et al. 2012) would need one
additional constrained optimization per prediction point.
"""

import re
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from ..model.solving import (model_ODE_solution, get_bacterial_count, observable_NGS, observable_MALDI,
                             observable_platecount)
from ..parameter_estimation.cost_function import ode_params_for_model, cost_initvals
from ..parameter_estimation.profile_likelihood import _read_csv_and_summary, rereference_to_best
from ..tools.dataframe_functions import extract_observables_from_df

OBSERVABLES = ('x', 'platecount', 'maldi', 'ngs')
_X0_NAME = re.compile(r'x_\{0,[^}]*\}\^\{([^}]*)\}')


# ----------------------------------------------------------------------
# 1) Parameter ensembles
# ----------------------------------------------------------------------

def threshold_value(confidence_level=0.95, dof=1, scale=1.0):
    """Cost difference threshold chi2.ppf(confidence_level, dof) * scale."""
    return float(chi2.ppf(confidence_level, dof) * scale)


def _exps_from_names(names):
    """Calibration experiments in the order of the x0 block, from names like '$x_{0,00}^{V01}$'."""
    if names is None:
        return None
    exps = []
    for n in names:
        m = _X0_NAME.search(str(n))
        if m and m.group(1) not in exps:
            exps.append(m.group(1))
    return exps or None


def _with_S_from_names(names):
    """True if the parameter names contain media coefficients S."""
    return None if names is None else any(str(n).startswith('$S_') or str(n).startswith('S_') for n in names)


def ensemble_from_fits(params, costs, cost_opt=None, param_opt=None, confidence_level=0.95, dof=1, scale=1.0,
                       names=None, exps=None, with_S=None, max_members=None, seed=0, unique=True):
    """
    Parameter ensemble from any set of fits: all vectors with cost - cost_opt <= threshold.

    params     : (n_fits, n_param) full parameter vectors
    costs      : (n_fits,) their costs (same objective and scale as the threshold)
    cost_opt   : reference cost; None: the lowest cost (also used if a fit is lower than cost_opt)
    param_opt  : the estimate; None: the fit with the lowest cost
    dof        : degrees of freedom of the chi2 threshold (1: as for the profile-likelihood CIs)
    scale      : converts cost differences to Delta(-2 log L), see estimate_profile_scale
    names      : parameter names (optional), used to infer exps and with_S
    exps       : calibration experiments in the order of the x0 block
    with_S     : vectors end with S (n_media x n_cl); None: from names, else decided in predict_ensemble
    max_members: keep at most this many members (random subset, the estimate is always kept)
    unique     : drop duplicate vectors (e.g. the optimum, contained once in every profile)
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    costs = np.asarray(costs, dtype=float).ravel()
    ok = np.isfinite(costs) & np.all(np.isfinite(params), axis=1)
    params, costs = params[ok], costs[ok]
    if len(costs) == 0:
        raise ValueError('ensemble_from_fits: no finite fits.')
    i_best = int(np.argmin(costs))
    if cost_opt is None or costs[i_best] < cost_opt:  # the best fit is the reference
        cost_opt, param_opt = float(costs[i_best]), params[i_best]
    elif param_opt is None:
        param_opt = params[i_best]
    thr = threshold_value(confidence_level, dof, scale)
    keep = costs - cost_opt <= thr
    params, costs = params[keep], costs[keep]
    if unique and len(params):
        _, idx = np.unique(np.round(params, 12), axis=0, return_index=True)
        idx = np.sort(idx)
        params, costs = params[idx], costs[idx]
    if max_members is not None and len(params) > max_members:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(params), size=max_members, replace=False))
        params, costs = params[idx], costs[idx]
    return {
        'params': params,
        'costs': costs,
        'param_opt': np.asarray(param_opt, dtype=float),
        'cost_opt': float(cost_opt),
        'names': None if names is None else list(names),
        'exps': exps if exps is not None else _exps_from_names(names),
        'with_S': with_S if with_S is not None else _with_S_from_names(names),
        'threshold': thr,
        'confidence_level': confidence_level,
        'dof': dof,
        'scale': float(scale),
    }


def ensemble_from_profile_csv(csv_path, confidence_level=0.95, dof=1, scale='auto', param_opt=None,
                              profile_indices=None, exps=None, with_S=None, max_members=None, seed=0,
                              reference='best'):
    """
    Parameter ensemble from the CSV of a profile-likelihood run (or its _partial.csv while it runs):
    every profile point with Delta cost <= chi2.ppf(confidence_level, dof) * scale.

    scale          : 'auto' -> from <csv>_summary.json / _run.json (1 for -2 log L objectives)
    profile_indices: use only the profiles of these parameters (indices into the parameter vector)
    reference      : 'best' -> if a profile point is lower than the stored optimum, it becomes the reference
                     (as in plot_profile_likelihood_from_csv); 'stored' -> always the stored optimum
    exps, with_S   : None -> inferred from the column names ($x_{0,..}^{V01}$, $S_{..}$)
    """
    df, names, scale, cost_opt, p_opt = _read_csv_and_summary(csv_path, scale, param_opt=param_opt)
    if reference == 'best':
        df, cost_opt, p_opt, _ = rereference_to_best(df, names, cost_opt, p_opt, scale=scale,
                                                     confidence_level=confidence_level)
    if profile_indices is not None:
        df = df[df['param_index'].isin(list(profile_indices))]
    params = df[names].to_numpy(dtype=float)
    ens = ensemble_from_fits(np.vstack([params, p_opt]), np.concatenate([df['cost'].to_numpy(float), [cost_opt]]),
                             cost_opt=cost_opt, param_opt=p_opt, confidence_level=confidence_level, dof=dof,
                             scale=scale, names=names, exps=exps, with_S=with_S, max_members=max_members, seed=seed)
    ens['source'] = str(csv_path)
    return ens


def ensemble_summary(ensemble):
    """Per-parameter range covered by the ensemble (DataFrame: estimate, min, max)."""
    names = ensemble['names'] or [f'p{i}' for i in range(ensemble['params'].shape[1])]
    p = ensemble['params']
    return pd.DataFrame({'estimate': ensemble['param_opt'], 'min': p.min(axis=0), 'max': p.max(axis=0)},
                        index=names)


# ----------------------------------------------------------------------
# 2) Simulation of the ensemble
# ----------------------------------------------------------------------

def split_param(param, setup, n_x0_exps, with_S):
    """Full parameter vector -> (x0 block (n_x0_exps, n_cl), raw ODE parameters, s_x (n_media, n_cl))."""
    n_cl, n_media = setup['n_cl'], setup['n_media']
    param = np.asarray(param, dtype=float)
    n_x0 = n_cl * n_x0_exps
    x0 = param[:n_x0].reshape((n_x0_exps, n_cl))
    if with_S:
        return x0, param[n_x0:-n_cl*n_media], param[-n_cl*n_media:].reshape((n_media, n_cl))
    return x0, param[n_x0:], np.asarray(setup['s_x'], dtype=float).reshape((n_media, n_cl))


def simulate_experiment(param_ode, s_x, x0_log10, temp, setup, t_model):
    """
    Model curves of one experiment: dict with 'x' (n_cl, T), 'platecount' (n_media, T),
    'maldi' (n_media, n_cl, T), 'ngs' (n_cl, T). param_ode as in the parameter vector
    (ode_params_for_model is applied here). NaN if the ODE solve fails.
    """
    n_cl = setup['n_cl']
    t_model = np.asarray(t_model, dtype=float)
    const = [[temp], n_cl, setup['media']]
    C0 = np.concatenate((10**np.asarray(x0_log10, dtype=float), np.ones(n_cl + 1)))
    nan = {'x': np.full((n_cl, len(t_model)), np.nan),
           'platecount': np.full((setup['n_media'], len(t_model)), np.nan),
           'maldi': np.full((setup['n_media'], n_cl, len(t_model)), np.nan),
           'ngs': np.full((n_cl, len(t_model)), np.nan)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C = model_ODE_solution(setup['model'], t_model, ode_params_for_model(param_ode, setup), C0, const)
            if np.shape(C)[-1] != len(t_model) or not np.all(np.isfinite(C)):
                return nan
            n_C = get_bacterial_count(C, n_cl, 2)
            n_C0 = get_bacterial_count(C0.reshape(len(C0), 1), n_cl, 2)
            return {'x': n_C,
                    'platecount': observable_platecount(t_model, n_C, s_x, n_C0, const, t_model)[1],
                    'maldi': observable_MALDI(t_model, n_C, s_x, n_C0, const, t_model)[1],
                    'ngs': observable_NGS(t_model, n_C, setup['T_x'], n_C0, const, t_model)[1]}
    except Exception:
        return nan


def refit_initial_values(param_ode, s_x, setup, x0_start, x0_bnds=None, maxiter=200):
    """
    Re-estimate the log10 initial values of the experiments in setup['dfs'] for fixed ODE parameters
    and S (cost_initvals, L-BFGS-B from x0_start). Returns (x0 vector in sorted experiment order, cost).
    setup needs 'data_array' (extract_observables_from_df(setup['dfs'])).
    """
    s = dict(setup, param_ode=np.asarray(param_ode, dtype=float), s_x=np.asarray(s_x, dtype=float))
    bnds = x0_bnds if x0_bnds is not None else setup.get('param_bnds')
    x0_start = np.asarray(x0_start, dtype=float)
    if bnds is not None:
        lo, hi = np.array(bnds, dtype=float).T
        x0_start = np.clip(x0_start, lo, hi)

    def f(x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                c = float(cost_initvals(x, s, None))
            except Exception:
                return 1e10
        return c if np.isfinite(c) else 1e10

    res = minimize(f, x0_start, method='L-BFGS-B', bounds=bnds, options={'maxiter': maxiter})
    return res.x, float(res.fun)


def _member_prediction(param, setup, t_model, exps, exp_index, x0_fixed, with_S, n_x0_exps,
                       refit, fit_exps, x0_bnds):
    """Predictions of one ensemble member for all exps: ({exp: {obs: array}}, {exp: x0})."""
    x0_block, param_ode, s_x = split_param(param, setup, n_x0_exps, with_S)
    x0_member = {}
    if refit:
        start = np.concatenate([x0_fixed[e] for e in fit_exps])
        x0_vec, _ = refit_initial_values(param_ode, s_x, setup, start, x0_bnds)
        n_cl = setup['n_cl']
        x0_member = {e: x0_vec[n_cl*i:n_cl*(i+1)] for i, e in enumerate(fit_exps)}
    out, x0_used = {}, {}
    for exp in exps:
        if exp in x0_member:
            x0 = x0_member[exp]
        elif x0_fixed is not None and exp in x0_fixed:
            x0 = np.asarray(x0_fixed[exp], dtype=float)
        else:
            x0 = x0_block[exp_index[exp]]
        x0_used[exp] = x0
        out[exp] = simulate_experiment(param_ode, s_x, x0, setup['exp_temps'][exp], setup, t_model)
    return out, x0_used


_POOL = {}


def _pool_init(kwargs):
    _POOL.clear()
    _POOL.update(kwargs)


def _pool_task(param):
    return _member_prediction(param, **_POOL)


def predict_ensemble(ensemble, setup, t_model, exps=None, x0=None, refit_x0=False, x0_bnds=None,
                     calibr_exps=None, with_S=None, include_estimate=True, n_jobs=1, verbose=True):
    """
    Simulate every member of the ensemble.

    ensemble    : from ensemble_from_profile_csv / ensemble_from_fits
    setup       : calibration or prediction setup ('model', 'n_cl', 'n_media', 'media', 'T_x', 'exp_temps',
                  's_x' if the vectors contain no S, 'dfs' for refit_x0 and for the data in the figures)
    t_model     : time points of the curves (starting at >= 0)
    exps        : experiments to predict; None: the calibration experiments of the ensemble
    x0          : {exp: log10 initial values} for experiments that are not in the x0 block of the vectors
                  (new experiments); for calibration experiments it overrides the vectors' x0
    refit_x0    : re-estimate x0 of the experiments in setup['dfs'] for each member (start: x0)
    x0_bnds     : bounds of that fit (n_cl per experiment, sorted experiment order); default setup['param_bnds']
    calibr_exps : experiments of the x0 block; None: ensemble['exps'], else setup['exps']
    with_S      : vectors end with S; None: ensemble['with_S'], else inferred from the vector length
    include_estimate: also simulate ensemble['param_opt'] (with the given x0 / its own refit)
    n_jobs      : processes over the members (-1: all cores; spawned processes, so the calling script
                  needs an  if __name__ == "__main__":  guard)

    Returns dict: 't', 'exps', 'members' {exp: {obs: (n_members, ...)}}, 'estimate' {exp: {obs: ...}} or None,
    'x0' {exp: (n_members, n_cl)}, 'x0_estimate', 'n_failed', and the ensemble information.
    """
    t_model = np.asarray(t_model, dtype=float)
    n_cl, n_media = setup['n_cl'], setup['n_media']
    calibr_exps = calibr_exps or ensemble.get('exps') or list(setup['exps'])
    exps = list(exps) if exps is not None else list(calibr_exps)
    exp_index = {e: i for i, e in enumerate(calibr_exps)}
    n_x0_exps = len(calibr_exps)
    n_param = ensemble['params'].shape[1]
    if with_S is None:
        with_S = ensemble.get('with_S')
    if with_S is None:
        n_ode = n_param - n_cl * n_x0_exps
        with_S = 's_x' not in setup or setup.get('s_x') is None
        if with_S:
            n_ode -= n_cl * n_media
        if verbose:
            print(f'predict_ensemble: with_S = {with_S} ({n_ode} ODE parameters)')

    x0 = None if x0 is None else {e: np.asarray(v, dtype=float) for e, v in x0.items()}
    missing = [e for e in exps if e not in exp_index and (x0 is None or e not in x0)]
    if missing:
        raise ValueError(f'predict_ensemble: no initial values for {missing}; pass x0={{exp: log10 x0}} '
                         f'(the x0 block of the vectors has {calibr_exps}).')

    fit_exps = None
    if refit_x0:
        if 'dfs' not in setup:
            raise ValueError("refit_x0=True needs the data of the experiments in setup['dfs'].")
        fit_exps = sorted(set(s.split('_')[0] for s in setup['dfs'][0].columns))
        if x0 is None or any(e not in x0 for e in fit_exps):
            raise ValueError(f'refit_x0=True needs start values x0 for all experiments in the data: {fit_exps}')
        bnds = x0_bnds if x0_bnds is not None else setup.get('param_bnds')
        if bnds is not None and len(bnds) != n_cl * len(fit_exps):
            raise ValueError(f'x0_bnds: {len(bnds)} bounds for {len(fit_exps)} experiments x {n_cl} species.')
        setup = dict(setup, exps=fit_exps, data_array=extract_observables_from_df(setup['dfs']))
        x0_bnds = bnds

    kwargs = dict(setup=setup, t_model=t_model, exps=exps, exp_index=exp_index, x0_fixed=x0, with_S=with_S,
                  n_x0_exps=n_x0_exps, refit=bool(refit_x0), fit_exps=fit_exps, x0_bnds=x0_bnds)
    params = ensemble['params']
    if verbose:
        print(f'predict_ensemble: {len(params)} members, experiments {exps}'
              + (', x0 refitted per member' if refit_x0 else ''), flush=True)
    if n_jobs is not None and n_jobs != 1 and len(params) > 1:
        n_proc = mp.cpu_count() if n_jobs == -1 else int(n_jobs)
        with mp.get_context('spawn').Pool(n_proc, initializer=_pool_init, initargs=(kwargs,)) as pool:
            results = pool.map(_pool_task, list(params), chunksize=max(1, len(params) // (4 * n_proc)))
    else:
        results = [_member_prediction(p, **kwargs) for p in params]

    members, x0_all = {}, {}
    ok = np.array([all(np.all(np.isfinite(r[0][e]['x'])) for e in exps) for r in results], dtype=bool)
    for e in exps:
        members[e] = {o: np.array([r[0][e][o] for r, k in zip(results, ok) if k]) for o in OBSERVABLES}
        x0_all[e] = np.array([r[1][e] for r, k in zip(results, ok) if k])
    n_failed = int((~ok).sum())
    if n_failed and verbose:
        print(f'predict_ensemble: {n_failed} members dropped (ODE solve failed)')

    estimate, x0_est = None, None
    if include_estimate:
        est, x0_est = _member_prediction(ensemble['param_opt'], **kwargs)
        estimate = est

    return {'t': t_model, 'exps': exps, 'members': members, 'estimate': estimate, 'x0': x0_all,
            'x0_estimate': x0_est, 'n_members': int(ok.sum()), 'n_failed': n_failed, 'media': list(setup['media']),
            'confidence_level': ensemble.get('confidence_level'), 'dof': ensemble.get('dof'),
            'refit_x0': bool(refit_x0), 'band_method': ensemble.get('band_method', 'envelope')}


# ----------------------------------------------------------------------
# 3) Bands
# ----------------------------------------------------------------------

def prediction_bands(prediction, method=None, quantiles=(0.025, 0.975), include_estimate=True):
    """
    Pointwise prediction band of every experiment and observable: {exp: {obs: (lower, upper)}}.
    method: 'envelope' (min / max over the members; for profile / threshold ensembles) or
            'quantile' (quantiles over the members; for samples of a distribution, e.g. fm.mcmc);
            None: prediction['band_method'] (from the ensemble: 'quantile' for MCMC, else 'envelope').
    include_estimate: the band also contains the estimate (envelope only).
    """
    method = method or prediction.get('band_method', 'envelope')
    bands = {}
    for e in prediction['exps']:
        bands[e] = {}
        for o in OBSERVABLES:
            m = prediction['members'][e][o]
            if include_estimate and method == 'envelope' and prediction.get('estimate') is not None:
                m = np.concatenate([m, prediction['estimate'][e][o][None]], axis=0)
            if len(m) == 0:
                bands[e][o] = (None, None)
                continue
            if method == 'envelope':
                bands[e][o] = (np.nanmin(m, axis=0), np.nanmax(m, axis=0))
            elif method == 'quantile':
                lo, hi = np.nanquantile(m, quantiles, axis=0)
                bands[e][o] = (lo, hi)
            else:
                raise ValueError("method must be 'envelope' or 'quantile'")
    return bands


def prediction_to_dataframe(prediction, bands=None, species=None, **band_kwargs):
    """
    Long table of estimate and band: columns exp, observable, medium, species, t, estimate, lower, upper.
    species: names of the species (default 0, 1, ...). Write it with .to_csv(...).
    """
    bands = bands if bands is not None else prediction_bands(prediction, **band_kwargs)
    t = prediction['t']
    media = prediction['media']
    rows = []

    def add(exp, obs, med, sp, est, lo, hi):
        rows.append(pd.DataFrame({'exp': exp, 'observable': obs, 'medium': med, 'species': sp, 't': t,
                                  'estimate': np.nan if est is None else est, 'lower': lo, 'upper': hi}))

    for e in prediction['exps']:
        est = prediction['estimate'][e] if prediction.get('estimate') is not None else None
        b = bands[e]
        n_cl = b['x'][0].shape[0]
        sp_names = list(species) if species is not None else list(range(n_cl))
        for k in range(n_cl):
            add(e, 'x', '', sp_names[k], None if est is None else est['x'][k], b['x'][0][k], b['x'][1][k])
            add(e, 'ngs', '', sp_names[k], None if est is None else est['ngs'][k], b['ngs'][0][k], b['ngs'][1][k])
        for j, med in enumerate(media):
            add(e, 'platecount', med, '', None if est is None else est['platecount'][j],
                b['platecount'][0][j], b['platecount'][1][j])
            for k in range(n_cl):
                add(e, 'maldi', med, sp_names[k], None if est is None else est['maldi'][j, k],
                    b['maldi'][0][j, k], b['maldi'][1][j, k])
    return pd.concat(rows, ignore_index=True)

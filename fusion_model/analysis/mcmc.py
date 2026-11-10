"""
MCMC sampling of the model parameters with emcee (pip install emcee, or pip install -e ".[mcmc]")
===============================================================================================

Samples the parameter vectors in proportion to their likelihood,

    log p(theta | data) = -cost(theta) / (2 * scale) + const,   theta inside param_bnds (flat prior), else -inf,

with cost = -2 log L (scale = 1, e.g. GaussianNeg2LogLik / objective 'neg2logL'). Only the free parameters
(param_bnds with hi > lo, minus fixed_indices) are sampled; the others stay at their values in param_opt.

    res = fm.mcmc.run_mcmc(calibr_setup, param_opt, n_steps=5000, init_ensemble=ens_profile,
                           out_file=path + 'mcmc.npz')
    fm.mcmc.print_diagnostics(res)                       # acceptance, autocorrelation time, suggested burn-in
    fm.mcmc.plot_mcmc_traces(res, path + 'mcmc_')        # traces and marginal histograms
    ens = fm.mcmc.ensemble_from_mcmc(res, max_members=500)
    pred = fm.pred_unc.predict_ensemble(ens, calibr_setup, t)      # bands: quantiles over the samples
    fm.plotting.plot_prediction_uncertainty(pred, calibr_setup, path=path, add_name='_mcmc')

Notes
* emcee runs an ensemble of walkers (default 2 * n_free + 2). Moves: 80 % DEMove + 20 % DESnookerMove
  (differential evolution, efficient for correlated parameters); pass moves=... for others.
* Start: init_ensemble (e.g. fm.pred_unc.ensemble_from_profile_csv) gives starting points spread over the
  confidence region, which shortens the burn-in; otherwise a small ball around param_opt.
* Cost: backend='jax' uses the JAX cost (the same one as fits with backend='jax'), otherwise
  cost_withS / cost_direct. n_jobs > 1 evaluates the walkers in processes (the calling script needs an
  if __name__ == "__main__": guard).
* Convergence: run at least ~50 autocorrelation times; discard ~2-3 tau as burn-in (print_diagnostics).
* The samples are a Bayesian posterior with a flat prior on the bounds; the bands are credible bands. For
  well-identified parameters they agree with the confidence bands; for poorly identified parameters the
  bounds (the prior) matter. With GaussianNeg2LogLik the noise variances are profiled out, not sampled.
"""
import os
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from ..parameter_estimation.profile_likelihood import free_param_indices
from ..parameter_estimation.cost_function import cost_withS, cost_direct

PENALTY = 1e10  # cost returned by the cost functions when the ODE solve fails


def _emcee():
    try:
        import emcee
    except ImportError as e:
        raise ImportError('fm.mcmc needs emcee: pip install emcee  (or pip install -e ".[mcmc]")') from e
    return emcee


def default_cost_func(backend='jax', with_S=True):
    """cost_withS / cost_direct, with the JAX backend if backend == 'jax'."""
    if backend == 'jax':
        from ..parameter_estimation.jax_backend import cost_withS_jax, cost_direct_jax
        return cost_withS_jax if with_S else cost_direct_jax
    return cost_withS if with_S else cost_direct


class LogPosterior:
    """log posterior of the free parameters (picklable, for process pools)."""

    def __init__(self, cost_func, setup, param_template, free_idx, bounds, scale=1.0):
        self.cost_func = cost_func
        self.setup = setup
        self.template = np.asarray(param_template, dtype=float)
        self.free_idx = np.asarray(free_idx, dtype=int)
        self.lo, self.hi = np.asarray(bounds, dtype=float).T
        self.scale = float(scale)

    def full(self, theta):
        p = self.template.copy()
        p[self.free_idx] = theta
        return p

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        if not np.all(np.isfinite(theta)) or np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        try:
            c = float(self.cost_func(self.full(theta), self.setup, None))
        except Exception:
            return -np.inf
        if not np.isfinite(c) or c >= 0.1 * PENALTY:
            return -np.inf
        return -0.5 * c / self.scale


_WORKER = {}


def _worker_init(log_post):
    """Pool initializer: the log posterior is sent once per process (the JAX cost is compiled once)."""
    from ..parameter_estimation.optimization import jax_worker_init
    jax_worker_init()
    _WORKER['log_post'] = log_post


def _worker_log_post(theta):
    return _WORKER['log_post'](theta)


def initial_positions(log_post, n_walkers, theta_opt, init_ensemble=None, jitter=1e-3, seed=0, max_tries=20):
    """
    Starting points of the walkers (n_walkers, n_free), all with a finite log posterior.
    init_ensemble: ensemble dict (params of full vectors) -> random members + small noise
                   (jitter * spread of the members per parameter); otherwise theta_opt + jitter * (hi - lo).
    """
    rng = np.random.default_rng(seed)
    lo, hi = log_post.lo, log_post.hi
    width = hi - lo
    if init_ensemble is not None:
        pool = np.asarray(init_ensemble['params'], dtype=float)[:, log_post.free_idx]
        spread = np.maximum(pool.std(axis=0), 1e-6 * np.maximum(width, 1e-12))
    else:
        pool, spread = theta_opt[None, :], width

    def draw(n):
        base = pool[rng.choice(len(pool), size=n, replace=n > len(pool))]
        x = base + jitter * spread * rng.standard_normal(base.shape)
        eps = 1e-9 * np.maximum(width, 1e-12)
        return np.clip(x, lo + eps, hi - eps)

    pos = draw(n_walkers)
    for _ in range(max_tries):
        bad = np.array([not np.isfinite(log_post(x)) for x in pos])
        if not bad.any():
            return pos
        pos[bad] = draw(int(bad.sum()))
    raise RuntimeError(f'initial_positions: {int(bad.sum())} walkers without finite log posterior; '
                       'check param_opt / init_ensemble / bounds.')


def default_moves():
    emcee = _emcee()
    return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


def _save(out_file, res):
    tmp = out_file + '.tmp.npz'
    np.savez(tmp, **{k: (np.asarray(v, dtype=object) if k == 'names' and v is not None else v)
                     for k, v in res.items() if v is not None})
    os.replace(tmp, out_file)


def load_mcmc(out_file):
    """Result of run_mcmc from its .npz file."""
    d = np.load(out_file, allow_pickle=True)
    res = {k: d[k] for k in d.files}
    for k in ('scale', 'n_walkers', 'with_S'):
        if k in res:
            res[k] = res[k].item()
    if 'names' in res:
        res['names'] = list(res['names'])
    return res


def run_mcmc(setup, param_opt, n_steps=5000, n_walkers=None, cost_func=None, backend='jax', with_S=True,
             fixed_indices=(), init_ensemble=None, scale=1.0, n_jobs=1, moves=None, out_file=None,
             checkpoint_every=200, resume=False, stop_file=None, seed=0, names=None, progress=True):
    """
    Run emcee for the free parameters of a calibration.

    setup        : calibration setup (with 'param_bnds' and 'data_array', and the objective as aggregation_func)
    param_opt    : estimate (full vector); values of the fixed parameters, start without init_ensemble
    n_steps      : steps (per walker) of this call; with resume=True they are added to the saved chain
    n_walkers    : default 2 * n_free + 2
    cost_func    : default cost_withS(_jax) / cost_direct(_jax) (backend, with_S)
    init_ensemble: ensemble dict for the starting points (e.g. from ensemble_from_profile_csv)
    scale        : cost -> -2 log L (1 for -2 log L objectives)
    out_file     : .npz written every checkpoint_every steps and at the end (needed for resume)
    stop_file    : if this file exists, the run stops at the next checkpoint (result saved)
    names        : parameter names (full vector), stored with the result

    Returns dict: chain (n_steps, n_walkers, n_free), log_prob (n_steps, n_walkers), acceptance (n_walkers,),
    free_idx, template (full vector), names, scale, with_S, time_s.
    """
    emcee = _emcee()
    cost_func = cost_func or default_cost_func(backend, with_S)
    param_opt = np.asarray(param_opt, dtype=float)
    bnds = setup['param_bnds']
    free_idx = free_param_indices(bnds, fixed_indices)
    fixed_set = set(int(i) for i in fixed_indices or ())
    template = param_opt.copy()
    for j in fixed_set:  # fixed parameters: their (equal) bounds if they define the value, else param_opt
        lo, hi = bnds[j]
        if lo == hi:
            template[j] = lo
    log_post = LogPosterior(cost_func, setup, template, free_idx, [bnds[j] for j in free_idx], scale)
    n_free = len(free_idx)
    n_walkers = int(n_walkers or (2 * n_free + 2))
    n_walkers += n_walkers % 2

    old = None
    if resume and out_file and os.path.exists(out_file):
        old = load_mcmc(out_file)
        if list(old['free_idx']) != list(free_idx) or old['chain'].shape[1] != n_walkers:
            raise ValueError('resume: the saved chain has other free parameters or another number of walkers.')
        p0 = old['chain'][-1]
        print(f"run_mcmc: resuming from {out_file} ({old['chain'].shape[0]} steps)", flush=True)
    else:
        p0 = initial_positions(log_post, n_walkers, param_opt[free_idx], init_ensemble, seed=seed)

    print(f'run_mcmc: {n_free} free parameters, {n_walkers} walkers, {n_steps} steps, '
          f'{"JAX" if backend == "jax" else "numpy"} cost, {n_jobs} process(es)', flush=True)
    pool, lp_func = None, log_post
    if n_jobs is not None and n_jobs != 1:
        n_proc = mp.cpu_count() if n_jobs == -1 else int(n_jobs)
        pool = mp.get_context('spawn').Pool(n_proc, initializer=_worker_init, initargs=(log_post,))
        lp_func = _worker_log_post
    sampler = emcee.EnsembleSampler(n_walkers, n_free, lp_func, moves=moves or default_moves(), pool=pool)
    # reproducible moves; a resumed run continues with another seed
    sampler.random_state = np.random.RandomState(seed + (0 if old is None else old['chain'].shape[0])).get_state()

    def result(t_run):
        chain, lp = sampler.get_chain(), sampler.get_log_prob()
        n_acc = sampler.acceptance_fraction * sampler.iteration
        n_tot = sampler.iteration
        if old is not None:
            chain = np.concatenate([old['chain'], chain])
            lp = np.concatenate([old['log_prob'], lp])
            n_old = old['chain'].shape[0]
            n_acc = n_acc + old['acceptance'] * n_old
            n_tot = n_tot + n_old
        return {'chain': chain, 'log_prob': lp, 'acceptance': n_acc / max(n_tot, 1),
                'free_idx': np.asarray(free_idx), 'template': template, 'names': names, 'scale': float(scale),
                'with_S': bool(with_S), 'n_walkers': n_walkers,
                'time_s': float(t_run + (old['time_s'] if old is not None and 'time_s' in old else 0.))}

    t0 = time.time()
    try:
        done = 0
        state = p0
        while done < n_steps:
            k = min(checkpoint_every, n_steps - done)
            state = sampler.run_mcmc(state, k, progress=progress)
            done += k
            res = result(time.time() - t0)
            if out_file:
                _save(out_file, res)
            m2_now = -2 * scale * sampler.get_log_prob()[-1]          # -2 log L of the walkers now
            m2_best = -2 * scale * np.max(res['log_prob'])            # best so far
            print(f'  step {done}/{n_steps}: acceptance {np.mean(sampler.acceptance_fraction):.2f}, '
                  f'best -2 log L {m2_best:.6g}, Delta of the walkers: median {np.median(m2_now - m2_best):.3g}, '
                  f'max {np.max(m2_now - m2_best):.3g}, {(time.time()-t0)/60:.1f} min', flush=True)
            if stop_file and os.path.exists(stop_file):
                print(f'run_mcmc: {stop_file} exists, stopping after {done} steps', flush=True)
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return result(time.time() - t0)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def autocorr_times(res, burn=0):
    """Integrated autocorrelation time of each free parameter (steps), emcee estimate without the
    length check (tol=0): reliable only if the chain is much longer (>~ 50 tau)."""
    emcee = _emcee()
    return emcee.autocorr.integrated_time(res['chain'][burn:], tol=0)


def print_diagnostics(res, burn=None):
    """Acceptance, autocorrelation times, suggested burn-in / thinning and the number of effective samples."""
    chain = res['chain']
    n_steps, n_walkers, _ = chain.shape
    tau = autocorr_times(res, burn or 0)
    tau_max = float(np.nanmax(tau))
    burn_s = int(np.ceil(3 * tau_max)) if burn is None else burn
    thin_s = max(1, int(np.floor(0.5 * np.nanmin(tau))))
    acc = res['acceptance']
    names = res.get('names')
    worst = np.argsort(tau)[::-1][:5]
    label = (lambda j: names[res['free_idx'][j]]) if names else (lambda j: f'p{res["free_idx"][j]}')
    print(f'MCMC: {n_steps} steps x {n_walkers} walkers, acceptance {np.mean(acc):.2f} '
          f'(min {np.min(acc):.2f}, max {np.max(acc):.2f})')
    print(f'  autocorrelation time: max {tau_max:.0f} steps, median {np.median(tau):.0f}; '
          f'chain length = {n_steps / tau_max:.1f} x max tau ' + ('(OK, > 50)' if n_steps > 50 * tau_max
                                                                    else '(too short for reliable estimates: run longer, resume=True)'))
    print('  slowest parameters: ' + ', '.join(f'{label(j)} ({tau[j]:.0f})' for j in worst))
    print(f'  suggested burn-in {burn_s} steps (3 x max tau), thinning {thin_s}; '
          f'effective samples ~ {max(0, n_steps - burn_s) * n_walkers / tau_max:.0f}')
    return {'tau': tau, 'burn': burn_s, 'thin': thin_s}


def _labels(res):
    names = res.get('names')
    return [names[j] if names else f'p{j}' for j in res['free_idx']]


def plot_mcmc_traces(res, file_prefix='mcmc_', burn=None, ncols=6, max_walkers=40, profile_ci=None):
    """
    <file_prefix>log_prob.png: Delta(-2 log L) of the walkers per step;
    <file_prefix>traces.png: trace of each free parameter (max_walkers walkers);
    <file_prefix>marginals.png: histogram of each parameter after the burn-in (vertical lines: profile CI
    from profile_ci = {full index: (lower, upper)}, e.g. from the profile summary JSON, optional).
    """
    chain, lp = res['chain'], res['log_prob']
    n_steps, n_walkers, n_free = chain.shape
    burn = print_diagnostics(res)['burn'] if burn is None else burn
    burn = min(burn, n_steps - 1)
    labels = _labels(res)
    w = np.arange(min(max_walkers, n_walkers))
    m2 = -2 * res['scale'] * lp
    ref = np.nanmin(m2[np.isfinite(m2)])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m2[:, w] - ref, lw=0.4, alpha=0.5)
    ax.axvline(burn, color='k', ls='--', lw=1)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('step')
    ax.set_ylabel(r'$\Delta(-2\log L)$')
    plt.savefig(file_prefix + 'log_prob.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    nrows = int(np.ceil(n_free / ncols))
    for kind in ('traces', 'marginals'):
        fig, axs = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.9 * nrows), squeeze=False)
        for j in range(n_free):
            ax = axs[j // ncols, j % ncols]
            if kind == 'traces':
                ax.plot(chain[:, w, j], lw=0.3, alpha=0.5)
                ax.axvline(burn, color='k', ls='--', lw=0.8)
            else:
                ax.hist(chain[burn:, :, j].ravel(), bins=40, color='gray', alpha=0.7)
                ax.axvline(res['template'][res['free_idx'][j]], color='k', lw=1)
                if profile_ci and int(res['free_idx'][j]) in profile_ci:
                    for v in profile_ci[int(res['free_idx'][j])]:
                        if v is not None:
                            ax.axvline(v, color='r', ls='--', lw=1)
                ax.set_yticks([])
            ax.set_title(labels[j], fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n_free, nrows * ncols):
            axs[j // ncols, j % ncols].axis('off')
        fig.tight_layout()
        plt.savefig(file_prefix + kind + '.png', bbox_inches='tight', dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------
# Samples -> ensemble for fm.pred_unc.predict_ensemble
# ----------------------------------------------------------------------

def samples(res, burn=None, thin=None):
    """Flattened samples of the full parameter vector after burn-in and thinning: (params, costs)."""
    if burn is None or thin is None:
        d = print_diagnostics(res)
        burn = d['burn'] if burn is None else burn
        thin = d['thin'] if thin is None else thin
    chain = res['chain'][burn::thin]
    lp = res['log_prob'][burn::thin]
    theta = chain.reshape(-1, chain.shape[-1])
    lp = lp.ravel()
    ok = np.isfinite(lp)
    params = np.repeat(res['template'][None, :], ok.sum(), axis=0)
    params[:, res['free_idx']] = theta[ok]
    return params, -2 * res['scale'] * lp[ok]


def ensemble_from_mcmc(res, burn=None, thin=None, max_members=500, seed=0, param_opt=None, exps=None):
    """
    Ensemble (as fm.pred_unc.ensemble_from_fits) of MCMC samples, at most max_members (random subset).
    Its prediction bands are quantiles over the samples (band_method 'quantile', default 95 %).
    param_opt: estimate for the curves of the estimate (default: the template = param_opt of run_mcmc).
    """
    from .prediction_uncertainty import _exps_from_names, _with_S_from_names
    params, costs = samples(res, burn, thin)
    if max_members is not None and len(params) > max_members:
        idx = np.sort(np.random.default_rng(seed).choice(len(params), size=max_members, replace=False))
        params, costs = params[idx], costs[idx]
    names = res.get('names')
    return {
        'params': params, 'costs': costs,
        'param_opt': np.asarray(res['template'] if param_opt is None else param_opt, dtype=float),
        'cost_opt': float(np.min(costs)), 'names': names,
        'exps': exps if exps is not None else _exps_from_names(names),
        'with_S': res.get('with_S', _with_S_from_names(names)),
        'threshold': None, 'confidence_level': 0.95, 'dof': None, 'scale': res['scale'],
        'band_method': 'quantile', 'source': 'mcmc',
    }

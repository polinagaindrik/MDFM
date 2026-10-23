"""
Least-squares local fit (trust-region reflective) for the calibration and profile costs
=====================================================================================

Alternative to L-BFGS-B for ill-conditioned problems: scipy.optimize.least_squares
(method='trf', a bounded Gauss-Newton / Levenberg-Marquardt type solver) works on the
residual vector and its Jacobian, and typically needs tens instead of hundreds of
iterations in narrow valleys of the cost.

* Any aggregation of the per-data-type sums of squares SSR_k (cost_sum_and_geometric_mean,
  GaussianNeg2LogLik, GeometricMeanNeg2LogLik, ...) is handled by iterative reweighting:
  at a stationary point of f(SSR_1, ..., SSR_K), sum_k df/dSSR_k * grad SSR_k = 0, i.e. it is
  a stationary point of the weighted sum of squares sum_k w_k SSR_k with w_k = df/dSSR_k.
  The weights are recomputed after every least-squares solve until the cost no longer
  decreases (usually 2-4 outer iterations). The weights are obtained numerically from
  calibr_setup['aggregation_func'], so no special case per aggregation is needed.
* The free parameters are mapped to [0, 1] (bound-scaled); with log_scale='auto' every
  parameter with 0 < lower bound and upper/lower >= 100 (k_ij, alpha, S) is mapped on a
  log10 scale. Fixed parameters (lower == upper) and fixed_indices stay at their value.
* Jacobian by forward differences, evaluated in n_jobs processes. The x0 of different
  experiments only change the residuals of their own experiment, so the x0 of all
  experiments for one species are perturbed together: n_cl instead of n_cl * n_exps
  model evaluations for the x0 block (group_x0=True).
"""
import os
import signal
import time

import numpy as np
from scipy.optimize import least_squares

from .optimization import die_with_parent
from .cost_function import cost_withS, residuals_withS, TotalCostFromResiduals

RESIDUAL_FUNCS = {cost_withS: residuals_withS}  # cost function -> matching residual function
PENALTY_RESIDUAL = 1e3  # residual value used when the ODE solve fails


def residual_func_for(cost_func, residual_func=None):
    """Residual function belonging to cost_func (or the one passed explicitly)."""
    if residual_func is not None:
        return residual_func
    try:
        return RESIDUAL_FUNCS[cost_func]
    except (KeyError, TypeError):
        raise ValueError(f"No residual function known for {getattr(cost_func, '__name__', cost_func)}; "
                         f"pass residual_func=... (signed residuals, see cost_function.residuals_withS).")


# ----------------------------------------------------------------------
# Residual evaluation (serial or in a process pool), with a time limit
# ----------------------------------------------------------------------
_REVAL = {}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _reval_init(residual_func, calibr_setup, jac_spasity, timeout, die=True):
    if die:
        die_with_parent()
    _REVAL.update(residual_func=residual_func, calibr_setup=calibr_setup, jac_spasity=jac_spasity,
                  timeout=timeout)


def _reval_one(x):
    """(residual arrays, owner arrays) or None if the solve failed / timed out."""
    timeout = _REVAL["timeout"]
    use_alarm = bool(timeout) and hasattr(signal, "SIGALRM")
    old = None
    try:
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, float(timeout))
        res, owner = _REVAL["residual_func"](x, _REVAL["calibr_setup"], _REVAL["jac_spasity"])
        if not all(np.all(np.isfinite(r)) for r in res):
            res = [np.where(np.isfinite(r), r, PENALTY_RESIDUAL) for r in res]
        return res, owner
    except Exception:  # includes _Timeout
        return None
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


class ResidualEvaluator:
    """Evaluates residual_func for a list of full parameter vectors (pool if n_jobs > 1)."""
    def __init__(self, residual_func, calibr_setup, jac_spasity=None, n_jobs=1, timeout=None):
        import multiprocessing as mp
        self.n_jobs = int(n_jobs) if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
        self.args = (residual_func, calibr_setup, jac_spasity, timeout)
        self.pool = None
        if self.n_jobs > 1:
            self.pool = mp.Pool(processes=self.n_jobs, initializer=_reval_init, initargs=self.args)
        _reval_init(*self.args, die=False)

    def __call__(self, xs):
        if self.pool is not None and len(xs) > 1:
            return self.pool.map(_reval_one, xs, chunksize=1)
        return [_reval_one(x) for x in xs]

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is not None and self.pool is not None:
            self.pool.terminate()
            self.pool = None
        self.close()


# ----------------------------------------------------------------------
# Parameter transformation to [0, 1] (log10 for wide positive ranges)
# ----------------------------------------------------------------------
class _Transform:
    def __init__(self, lo, hi, log_scale="auto"):
        self.lo, self.hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if log_scale == "auto":
            self.log = (self.lo > 0) & (self.hi / np.where(self.lo > 0, self.lo, 1.) >= 100.)
        elif log_scale:
            self.log = np.asarray(log_scale, dtype=bool)
        else:
            self.log = np.zeros(len(self.lo), dtype=bool)
        self.tlo = np.where(self.log, np.log10(np.where(self.log, self.lo, 1.)), self.lo)
        self.thi = np.where(self.log, np.log10(np.where(self.log, self.hi, 1.)), self.hi)

    def to_u(self, z):
        z = np.clip(np.asarray(z, dtype=float), self.lo, self.hi)
        t = np.where(self.log, np.log10(np.where(self.log, z, 1.)), z)
        return np.clip((t - self.tlo) / (self.thi - self.tlo), 0., 1.)

    def to_z(self, u):
        t = self.tlo + np.clip(u, 0., 1.) * (self.thi - self.tlo)
        return np.clip(np.where(self.log, 10. ** t, t), self.lo, self.hi)

    def dz_du(self, u):
        """Derivative of to_z (per parameter)."""
        z = self.to_z(u)
        return np.where(self.log, z * np.log(10.), 1.) * (self.thi - self.tlo)


# ----------------------------------------------------------------------
# Weights of the reweighted least-squares problem
# ----------------------------------------------------------------------
def aggregation_weights(aggregation_func, residuals, rel_eps=1e-6):
    """(cost, w_k = d cost / d SSR_k) of aggregation_func for the squared residuals."""
    sq = [np.asarray(r, dtype=float) ** 2 for r in residuals]
    f0 = float(aggregation_func(sq))
    w = []
    for k in range(len(sq)):
        ssr = float(np.sum(sq[k]))
        if ssr <= 0:  # e.g. penalty group exactly 0: derivative by an absolute step
            if len(sq[k]) == 0:
                w.append(0.)
                continue
            delta = 1e-10
            sq_k = list(sq)
            sq_k[k] = sq[k].copy()
            sq_k[k][0] += delta
            w.append(max((float(aggregation_func(sq_k)) - f0) / delta, 0.))
            continue
        sq_k = list(sq)
        sq_k[k] = sq[k] * (1. + rel_eps)
        w.append((float(aggregation_func(sq_k)) - f0) / (rel_eps * ssr))
    w = np.array(w)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError(f"aggregation weights {w} not positive: the aggregation is not increasing in the "
                         f"sums of squares, least_squares cannot be used for it.")
    return f0, w


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
def least_squares_fit(cost_func, x_start, free_idx, calibr_setup, residual_func=None, jac_spasity=None,
                      n_jobs=1, evaluator=None, log_scale="auto", group_x0=True, maxiter=200, max_outer=8,
                      outer_rtol=1e-9, ftol=1e-12, xtol=1e-12, gtol=1e-10, fd_step=1e-7, solve_timeout=None,
                      label=None, print_every=10, history=None, backend="numpy"):
    """
    Minimize calibr_setup['aggregation_func'] of the residuals over the parameters free_idx
    (all others held at x_start) with scipy.optimize.least_squares (trf) and reweighting.

    cost_func      : cost function whose residuals are fitted (fm.pest.cost_withS); only used to
                     find the residual function (residual_func=None) and for the final check.
    maxiter        : max. residual evaluations per least-squares solve (scipy's max_nfev; the
                     Jacobian evaluations are not counted)
    max_outer      : max. number of reweighting steps
    n_jobs         : processes for the finite-difference Jacobian (ignored if evaluator is given)
    history        : list; (full parameter vector, cost) is appended at every Jacobian evaluation
    backend        : 'numpy' (finite-difference Jacobian, n_jobs processes) or 'jax' (exact Jacobian by
                     automatic differentiation, jax_backend.JaxCost; n_jobs and evaluator are not used)
    Returns (cost, full parameter vector, success, number of Jacobian evaluations).
    """
    residual_func = residual_func_for(cost_func, residual_func)
    # cost as a function of the residual groups (incl. the optional s_max penalty group of residuals_withS)
    agg = (TotalCostFromResiduals(calibr_setup) if residual_func is residuals_withS
           else calibr_setup["aggregation_func"])
    x_start = np.array(x_start, dtype=float)
    free_idx = np.asarray(free_idx, dtype=int)
    bnds = calibr_setup["param_bnds"]
    lo = np.array([bnds[j][0] for j in free_idx], dtype=float)
    hi = np.array([bnds[j][1] for j in free_idx], dtype=float)
    tr = _Transform(lo, hi, log_scale)
    tag = label or "least_squares"
    t0 = time.time()

    import multiprocessing as mp
    if mp.current_process().daemon:  # inside a pool worker: no nested pool
        n_jobs = 1
    jc = None
    if backend == "jax":
        from .jax_backend import jax_cost_for
        jc = jax_cost_for(calibr_setup)
        evaluator = lambda xs: [([np.where(np.isfinite(r), r, PENALTY_RESIDUAL) for r in jc.residuals(x)], None)  # noqa: E731
                                for x in xs]
    elif backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'jax'")
    own_eval = evaluator is None
    ev = evaluator if evaluator is not None else ResidualEvaluator(residual_func, calibr_setup, jac_spasity,
                                                                   n_jobs=n_jobs, timeout=solve_timeout)
    try:
        def full(u):
            x = x_start.copy()
            x[free_idx] = tr.to_z(u)
            return x

        base = ev([x_start])[0]
        if base is None:
            raise RuntimeError("model evaluation at the start point failed")
        n_rows = [len(r) for r in base[0]]
        owners = None if base[1] is None else np.concatenate(base[1])

        # Jacobian column groups: x0 of the same species in different experiments together
        groups = [[c] for c in range(len(free_idx))]
        if group_x0 and "n_cl" in calibr_setup and "exps" in calibr_setup:
            n_cl, n_exps = int(calibr_setup["n_cl"]), len(calibr_setup["exps"])
            by_species = {}
            singles = []
            for c, j in enumerate(free_idx):
                if j < n_cl * n_exps:
                    by_species.setdefault(j % n_cl, []).append(c)
                else:
                    singles.append([c])
            groups = list(by_species.values()) + singles
            col_owner = {c: (free_idx[c] // n_cl if free_idx[c] < n_cl * n_exps else None)
                         for c in range(len(free_idx))}
        else:
            col_owner = {c: None for c in range(len(free_idx))}

        state = {"w": None, "cache_u": None, "cache_r": None, "njev": 0, "best": (np.inf, x_start.copy())}

        def raw(u):
            if state["cache_u"] is not None and np.array_equal(u, state["cache_u"]):
                return state["cache_r"]
            out = ev([full(u)])[0]
            r = None if out is None else out[0]
            state["cache_u"], state["cache_r"] = np.array(u, copy=True), r
            return r

        def weighted(r_list):
            if r_list is None:
                return np.full(sum(n_rows), PENALTY_RESIDUAL)
            return np.concatenate([np.sqrt(wk) * r for wk, r in zip(state["w"], r_list)])

        def fun(u):
            return weighted(raw(u))

        def jac(u):
            r0 = raw(u)
            if jc is not None:  # exact Jacobian (JAX): d r / d x  *  d x / d u, rows weighted
                J = jc.residual_jacobian(full(u))[:, free_idx] * tr.dz_du(u)[None, :]
                J = np.where(np.isfinite(J), J, 0.)
                J = J * np.repeat(np.sqrt(state["w"]), n_rows)[:, None]
                return _jac_bookkeeping(u, r0, J)
            f0 = weighted(r0)
            pts, steps = [], []
            for g in groups:
                h = np.zeros(len(u))
                step = fd_step if np.all(u[g] + fd_step <= 1.) else -fd_step
                h[g] = step
                pts.append(full(u + h))
                steps.append(step)
            outs = ev(pts)
            J = np.zeros((len(f0), len(u)))
            for g, step, out in zip(groups, steps, outs):
                d = (weighted(None if out is None else out[0]) - f0) / step
                if len(g) == 1:
                    J[:, g[0]] = d
                else:
                    for c in g:
                        rows = owners == col_owner[c]
                        J[rows, c] = d[rows]
            return _jac_bookkeeping(u, r0, J)

        def _jac_bookkeeping(u, r0, J):
            state["njev"] += 1
            if r0 is not None:
                c_now = float(agg([r ** 2 for r in r0]))
                x_now = full(u)
                if history is not None:
                    history.append((x_now, c_now))
                if c_now < state["best"][0]:
                    state["best"] = (c_now, x_now)
                if print_every and state["njev"] % print_every == 0:
                    print(f"    {tag}: Jacobian {state['njev']}, cost = {c_now:.10g}, "
                          f"{time.time() - t0:.0f} s", flush=True)
            return J

        u = tr.to_u(x_start[free_idx])
        r_list = base[0]
        cost, success = np.inf, False
        for it in range(max_outer):
            if r_list is None:
                break
            c_cur, w = aggregation_weights(agg, r_list)
            state["w"] = w / np.max(w)  # only the ratios matter
            if c_cur < state["best"][0]:
                state["best"] = (c_cur, full(u))
            sol = least_squares(fun, u, jac=jac, bounds=(0., 1.), method="trf", x_scale="jac",
                                ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=maxiter)
            r_new = raw(sol.x)
            c_new = np.inf if r_new is None else float(agg([r ** 2 for r in r_new]))
            if print_every:
                print(f"    {tag}: reweighting step {it + 1}: cost {c_cur:.10g} -> {c_new:.10g} "
                      f"({sol.nfev} evaluations, {state['njev']} Jacobians so far; {sol.message})", flush=True)
            success = bool(sol.success)
            if not c_new < c_cur:
                break
            u, r_list = sol.x, r_new
            if c_new < state["best"][0]:
                state["best"] = (c_new, full(u))
            if abs(c_cur - c_new) <= outer_rtol * max(1., abs(c_new)):
                break
        cost, x_best = state["best"]
    finally:
        if own_eval and hasattr(ev, "close"):
            ev.close()
    return float(cost), x_best, success, state["njev"]


def latin_hypercube_starts(param_bnds, n_starts, seed=0, log_scale="auto", fixed_values=None):
    """n_starts full parameter vectors, Latin hypercube over the free parameters (log10 where
    log_scale applies, as in least_squares_fit); fixed parameters at lower bound or fixed_values."""
    lo = np.array([b[0] for b in param_bnds], dtype=float)
    hi = np.array([b[1] for b in param_bnds], dtype=float)
    free = np.where(hi > lo)[0]
    tr = _Transform(lo[free], hi[free], log_scale)
    rng = np.random.default_rng(seed)
    n = int(n_starts)
    u = (np.argsort(rng.random((n, len(free))), axis=0) + rng.random((n, len(free)))) / n
    base = lo.copy() if fixed_values is None else np.array(fixed_values, dtype=float)
    starts = []
    for k in range(n):
        x = base.copy()
        x[free] = tr.to_z(u[k])
        starts.append(x)
    return starts

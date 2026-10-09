"""
Profile likelihood for fusion_model / model_paper
=================================================

Same public API as ``pool_paper_casestudy/fusion_core/likelihood.py``
(function names, argument names and defaults, return values, CSV format), so
that calls, CSV files and plotting scripts are interchangeable between the two
projects:

    free_param_indices(param_bnds)
    profile_likelihood_for_param(cost_func, param_opt, param_index, calibr_setup, span, n_points,
                                 method, jac_spasity, n_jobs, per_point_workers, n_restarts, jitter_frac)
                                 -> (grid, profile_cost, profile_params)
    confidence_interval_from_profile(grid, profile_cost, cost_opt, confidence_level, dof, scale) -> (lo, hi)
    count_data_points(cost_func, param, calibr_setup, jac_spasity)
    estimate_profile_scale(cost_func, param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity)
                                 -> (scale, n_data, sigma_hat2)
    run_profile_likelihood_all(cost_func, param_opt, calibr_setup, jac_spasity, span, n_points, method,
                               confidence_level, scale, n_jobs, per_point_workers, n_restarts, jitter_frac,
                               out_csv, param_names, plot, plot_path, ncols) -> (df, ci_results)
    plot_profile_likelihood(df, param_opt, cost_opt, ci_results, confidence_level, scale,
                            param_names, save_path, ncols)
    plot_profile_likelihood_from_csv(csv_path, confidence_level, scale, save_path, ncols)
    plot_profile_likelihood_individual(csv_path, confidence_level, scale, out_dir, file_prefix,
                                       file_ext, figsize)

Grid (as in pool_paper): ``n_points`` values in [p - span*|p|, p + span*|p|],
clipped to the bounds, plus the optimum itself (full bound range if p == 0).
For positive parameters this is exactly pool_paper's p*(1 -/+ span); writing it
with |p| also works for negative parameters such as lambda_1 in [-7, -2].

Differences in *how* the points are computed (results are compatible):

* ``method="local"``: the grid is walked outward from the optimum (left and
  right separately) and every point is warm-started from the solution of its
  neighbour (Raue et al. 2009); additionally from param_opt (as pool_paper's
  restart 0, ``anchor_start=True``) and from ``n_restarts - 1`` random
  jitters. The best of these local L-BFGS-B fits is kept.
* ``method="global"``: differential_evolution per grid point, with the
  neighbouring solution included in the initial population;
  ``per_point_workers`` workers inside each point (see CAVEAT ON
  MULTIPROCESSING in the pool_paper module).
* ``n_jobs`` parallelizes over walks, i.e. (parameter, direction) pairs; when there
  are fewer walks than n_jobs (and for the refit with ``refine=True``) it instead
  parallelizes the finite-difference gradient inside each L-BFGS-B fit.

Optional extras (keyword arguments with defaults; pool_paper calls work unchanged):
  profile_indices, fixed_indices, refine, stop_factor, anchor_start,
  minimize_options, solve_timeout, resume, true_params, verbose

Objective and scale
-------------------
The chi2 threshold assumes cost = -2 log L. ``GaussianNeg2LogLik``
(``make_gaussian_neg2loglik``) gives exactly that for Gaussian noise with one
unknown variance per data type (NGS, MALDI, MiBi), profiled out analytically:

    -2 log L = sum_k n_k * log(SSR_k / n_k) + const

With this aggregation ``scale="auto"`` resolves to 1. For any other aggregation
(e.g. cost_sum_and_geometric_mean or cost_arithmetic_mean) ``scale="auto"``
uses pool_paper's estimate (cost_opt / n_data), which is only meaningful for a
mean squared residual -- see the CAVEAT ON THE CHI2 THRESHOLD there.
"""

import os
import json
import time
import signal
import threading
import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2

from .optimization import CostEvaluator, fd_fun_and_grad

PENALTY = 1e10  # returned when the ODE solve fails / times out / gives NaN


# ----------------------------------------------------------------------
# -2 log L aggregation (alternative to cost_sum_and_geometric_mean)
# ----------------------------------------------------------------------

class GaussianNeg2LogLik:
    """
    Aggregation function for ``calibr_setup['aggregation_func']``.

    J_vect : list of squared-residual arrays, one per data type, in the order
             produced by the cost function (cost_withS: [ngs, maldi, mibi]).
    n_data : number of measured (finite) data points per data type, same order.

    Returns sum_k n_k * log(SSR_k / n_k), i.e. -2 log L up to a constant for
    Gaussian noise with an unknown, separately estimated variance per data
    type. NaN residuals (missing data) are ignored; infinite ones give PENALTY.
    n_k is counted from the data, not from the residual arrays (cost_withS
    drops exact-zero residuals), so it stays constant during the fit.
    """

    def __init__(self, n_data):
        self.n_data = [int(n) for n in n_data]

    def __call__(self, J_vect):
        total = 0.
        for J, n in zip(J_vect, self.n_data):
            if n == 0:
                continue
            J = np.asarray(J, dtype=float)
            if np.any(np.isinf(J)):
                return PENALTY
            ssr = np.nansum(J)  # NaN entries = missing data points
            total += n * np.log(max(ssr, 1e-300) / n)
        return total


def make_gaussian_neg2loglik(data_array):
    """
    Build a GaussianNeg2LogLik from ``calibr_setup['data_array']``
    (= extract_observables_from_df(dfs) -> (days, [obs_mibi, obs_maldi, obs_ngs])).
    The residual order of cost_withS / squared_differences is [ngs, maldi, mibi].
    """
    _, (obs_mibi, obs_maldi, obs_ngs) = data_array
    n_data = [np.count_nonzero(np.isfinite(obs_ngs)),
              np.count_nonzero(np.isfinite(obs_maldi)),
              np.count_nonzero(np.isfinite(obs_mibi))]
    return GaussianNeg2LogLik(n_data)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _SolveTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Raises _SolveTimeout if the wrapped block runs longer than `seconds`
    (SIGALRM: Unix, main thread of a process only; no-op otherwise)."""
    usable = (seconds and seconds > 0 and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def _handler(signum, frame):
        raise _SolveTimeout(f"solve exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(np.ceil(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _safe_cost(cost_func, x_full, calibr_setup, jac_spasity, solve_timeout):
    try:
        with time_limit(solve_timeout):
            c = float(cost_func(x_full, calibr_setup, jac_spasity))
    except Exception:
        return PENALTY
    return c if np.isfinite(c) else PENALTY


class _TimedCost:
    """cost_func with the solve timeout / failure handling of _safe_cost (picklable)."""
    def __init__(self, cost_func, solve_timeout):
        self.cost_func, self.solve_timeout = cost_func, solve_timeout

    def __call__(self, x, calibr_setup, jac_spasity):
        return _safe_cost(self.cost_func, x, calibr_setup, jac_spasity, self.solve_timeout)


def free_param_indices(param_bnds, fixed_indices=()):
    """Indices whose bounds are not fixed (lo == hi), excluding fixed_indices."""
    fixed = set(int(i) for i in (fixed_indices or ()))
    return [j for j, (lo, hi) in enumerate(param_bnds) if hi > lo and j not in fixed]


def _build_grid(param_opt, param_index, param_bnds, span, n_points):
    """pool_paper's grid, written with |p| so that it also works for negative values."""
    p_opt_val = float(param_opt[param_index])
    lo, hi = param_bnds[param_index]
    if hi <= lo:
        raise ValueError(f"param_index {param_index} is fixed in param_bnds ({lo}, {hi}); nothing to profile.")
    if p_opt_val == 0:
        grid = np.linspace(lo, hi, n_points)
    else:
        grid = np.linspace(max(lo, p_opt_val - span * abs(p_opt_val)),
                           min(hi, p_opt_val + span * abs(p_opt_val)), n_points)
    # make sure the optimum itself is included so profile_cost has a true minimum
    return np.sort(np.unique(np.concatenate([grid, [p_opt_val]])))


# ----------------------------------------------------------------------
# Progress reporting
# ----------------------------------------------------------------------

def _fmt_time(seconds):
    seconds = int(round(max(0., seconds)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _Progress:
    """Counter of finished profile points shared by all worker processes (elapsed time, ETA)."""
    def __init__(self, total):
        self.total = int(total)
        self.count = mp.Value("i", 0)
        self.t0 = time.time()

    def add(self, n=1):
        with self.count.get_lock():
            self.count.value += n
            done = self.count.value
        elapsed = time.time() - self.t0
        eta = elapsed / done * (self.total - done) if done > 0 else float("nan")
        pct = 100. * done / self.total if self.total else 100.
        return (f"total {done}/{self.total} points ({pct:.1f}%), elapsed {_fmt_time(elapsed)}"
                + (f", ETA {_fmt_time(eta)}" if np.isfinite(eta) else ""))


def _fit_others(cost_func, x_start, opt_idx, calibr_setup, jac_spasity, *, method="local",
                extra_starts=(), n_restarts=1, jitter_frac=0.05, rng=None, per_point_workers=1,
                minimize_options=None, solve_timeout=30, evaluator=None, label=None, print_every=0):
    """
    Minimize cost_func over the entries opt_idx of the full parameter vector;
    all other entries are held at their value in x_start.
    method='local' : L-BFGS-B from x_start, from each vector in extra_starts and
                     from n_restarts - 1 random jitters of x_start; best is kept.
    method='global': differential_evolution (x_start in the initial population).
    evaluator      : CostEvaluator with a process pool -> the finite-difference gradient
                     of L-BFGS-B is evaluated in parallel (method='local').
    label, print_every : print the cost every print_every L-BFGS-B iterations (0: silent).
    Returns (best_cost, best_full_vector, success, n_iterations).
    """
    x_start = np.asarray(x_start, dtype=float)
    opt_idx = np.asarray(opt_idx, dtype=int)
    if len(opt_idx) == 0:
        return _safe_cost(cost_func, x_start, calibr_setup, jac_spasity, solve_timeout), x_start.copy(), True, 0

    bnds = [calibr_setup["param_bnds"][j] for j in opt_idx]
    lo = np.array([b[0] for b in bnds], dtype=float)
    hi = np.array([b[1] for b in bnds], dtype=float)
    rng = rng if rng is not None else np.random.default_rng()

    def f(z):
        x = x_start.copy()
        x[opt_idx] = z
        return _safe_cost(cost_func, x, calibr_setup, jac_spasity, solve_timeout)

    if method == "global":
        res = differential_evolution(f, list(zip(lo, hi)), x0=np.clip(x_start[opt_idx], lo, hi),
                                     tol=1e-6, atol=1e-6, maxiter=100, popsize=15, polish=True,
                                     init="latinhypercube", updating="deferred", workers=per_point_workers)
        best_c, best_z, best_ok, nit = float(res.fun), res.x, bool(res.success), int(res.nit)
    elif method == "local":
        opts = {"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8}
        opts.update(minimize_options or {})
        # finite-difference gradients cost one evaluation per parameter; scipy's default
        # maxfun=15000 would otherwise stop large fits long before maxiter
        opts.setdefault("maxfun", 2 * (len(opt_idx) + 1) * opts["maxiter"])
        starts = [x_start[opt_idx]] + [np.asarray(xs, dtype=float)[opt_idx] for xs in extra_starts]
        for _ in range(max(0, int(n_restarts) - 1)):
            starts.append(x_start[opt_idx] + rng.normal(0., jitter_frac, size=len(opt_idx)) * (hi - lo))
        best_c, best_z, best_ok, nit = np.inf, starts[0], False, 0
        fg = None if evaluator is None else fd_fun_and_grad(evaluator, x_start, opt_idx, lo, hi)
        t0 = time.time()
        def _make_callback(r):
            # scipy passes the OptimizeResult only to callbacks whose single argument is
            # named `intermediate_result`, hence the factory instead of default arguments
            it = [0]
            start_txt = f" start {r + 1}/{len(starts)}" if len(starts) > 1 else ""

            def callback(intermediate_result):
                it[0] += 1
                if print_every and it[0] % print_every == 0:
                    print(f"    {label or 'fit'}{start_txt}: iteration {it[0]}, cost = {intermediate_result.fun:.8g}, "
                          f"{_fmt_time(time.time() - t0)}", flush=True)
            return callback

        for r, z0 in enumerate(starts):
            callback = _make_callback(r)

            if fg is None:
                res = minimize(f, np.clip(z0, lo, hi), method="L-BFGS-B", bounds=list(zip(lo, hi)), options=opts,
                               callback=callback)
            else:
                res = minimize(fg, np.clip(z0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                               options=opts, callback=callback)
            nit += int(res.nit)
            if res.fun < best_c:
                best_c, best_z, best_ok = float(res.fun), res.x, bool(res.success)
    else:
        raise ValueError("method must be 'local' or 'global'")

    x_best = x_start.copy()
    x_best[opt_idx] = best_z
    return best_c, x_best, best_ok, nit


def refine_optimum(cost_func, param_opt, calibr_setup, jac_spasity=None, fixed_indices=(),
                   n_restarts=1, jitter_frac=0.02, seed=0, minimize_options=None, solve_timeout=60, n_jobs=1,
                   print_every=10):
    """
    Local L-BFGS-B polish of param_opt with the *same* objective that is used
    for profiling. Needed when the profiling objective differs from the
    calibration one (GaussianNeg2LogLik vs. cost_sum_and_geometric_mean) or the
    calibration did not end exactly at a minimum; otherwise Delta becomes
    negative near the estimate.
    n_jobs > 1: the finite-difference gradient is evaluated in n_jobs processes.
    print_every: print the cost every print_every iterations (0: silent).
    """
    param_opt = np.asarray(param_opt, dtype=float)
    opt_idx = free_param_indices(calibr_setup["param_bnds"], fixed_indices)
    c0 = _safe_cost(cost_func, param_opt, calibr_setup, jac_spasity, solve_timeout)
    maxiter = (minimize_options or {}).get("maxiter", 300)
    print(f"refine_optimum: local re-fit of {len(opt_idx)} free parameters, start cost = {c0:.8g}, "
          f"max. {maxiter} iterations "
          f"({'parallel gradient in ' + str(n_jobs) + ' processes' if n_jobs > 1 else 'serial'}) ...", flush=True)
    t0 = time.time()
    with CostEvaluator(_TimedCost(cost_func, solve_timeout), calibr_setup, jac_spasity,
                       n_jobs=n_jobs if n_jobs > 1 else 1) as ev:
        c, x, ok, nit = _fit_others(cost_func, param_opt, opt_idx, calibr_setup, jac_spasity, method="local",
                                    n_restarts=n_restarts, jitter_frac=jitter_frac, rng=np.random.default_rng(seed),
                                    minimize_options=minimize_options, solve_timeout=solve_timeout,
                                    evaluator=ev if n_jobs > 1 else None, label="refine_optimum",
                                    print_every=print_every)
    if c > c0:
        c, x = c0, param_opt
    print(f"refine_optimum: cost {c0:.8g} -> {c:.8g} (success={ok}, {nit} iterations, "
          f"{_fmt_time(time.time() - t0)})", flush=True)
    return x, c


# ----------------------------------------------------------------------
# One outward walk (one parameter, one direction) on the grid
# ----------------------------------------------------------------------

def _profile_walk(cost_func, param_opt, cost_opt, idx, direction, grid, calibr_setup, jac_spasity, cfg, rng):
    """
    Re-optimize all other free parameters at the grid points on one side of the
    optimum, in order of increasing distance, warm-starting each point from the
    previous one. Returns a list of dicts.
    """
    p_hat = float(param_opt[idx])
    side = grid[grid > p_hat] if direction > 0 else grid[grid < p_hat][::-1]
    opt_idx = [j for j in cfg["free_idx"] if j != idx]
    threshold = chi2.ppf(cfg["confidence_level"], 1) * cfg["scale"]
    stop_delta = None if cfg["stop_factor"] is None else cfg["stop_factor"] * threshold

    name = cfg["param_names"][idx] if cfg.get("param_names") else f"param[{idx}]"
    tag = f"param[{idx}] {name} ({'+' if direction > 0 else '-'})"
    verbose = int(cfg["verbose"]) if cfg["verbose"] else 0
    progress = cfg.get("progress")

    x_prev = np.array(param_opt, dtype=float)
    rows = []
    for k, val in enumerate(side):
        t_point = time.time()
        x_start = x_prev.copy()
        x_start[idx] = val
        extra = []
        if cfg["anchor_start"] and k > 0:
            x_anchor = np.array(param_opt, dtype=float)
            x_anchor[idx] = val
            extra.append(x_anchor)
        c, x_new, ok, nit = _fit_others(cost_func, x_start, opt_idx, calibr_setup, jac_spasity,
                                        method=cfg["method"], extra_starts=extra, n_restarts=cfg["n_restarts"],
                                        jitter_frac=cfg["jitter_frac"], rng=rng,
                                        per_point_workers=cfg["per_point_workers"],
                                        minimize_options=cfg["minimize_options"],
                                        solve_timeout=cfg["solve_timeout"], evaluator=cfg.get("evaluator"),
                                        label=f"{tag} point {k + 1}/{len(side)}",
                                        print_every=10 if verbose >= 2 else 0)
        dt = time.time() - t_point
        rows.append({"param_index": idx, "direction": direction, "param_value": float(val),
                     "cost": c, "success": ok, "nit": nit, "time_s": dt, "x": x_new})
        x_prev = x_new
        stop = stop_delta is not None and (c - cost_opt) > stop_delta
        # points not computed because of the early stop count as done for the overall progress
        total_txt = progress.add(1 + (len(side) - k - 1 if stop else 0)) if progress is not None else ""
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] {tag} point {k + 1}/{len(side)}: value = {val:.6g}, "
                  f"cost = {c:.8g}, Delta = {(c - cost_opt) / cfg['scale']:.4g}, {nit} it., {_fmt_time(dt)}"
                  f"{'' if ok else ' (not converged)'}" + (f" | {total_txt}" if total_txt else ""), flush=True)
            if stop:
                print(f"  [{time.strftime('%H:%M:%S')}] {tag}: Delta > {cfg['stop_factor']} x threshold, "
                      f"skipping the remaining {len(side) - k - 1} points of this side", flush=True)
        if stop:
            break
    return rows


# --- multiprocessing plumbing (globals set once per worker) -----------
_G = {}


def _pool_init(cost_func, param_opt, cost_opt, grids, calibr_setup, jac_spasity, cfg):
    _G.update(cost_func=cost_func, param_opt=param_opt, cost_opt=cost_opt, grids=grids,
              calibr_setup=calibr_setup, jac_spasity=jac_spasity, cfg=cfg,
              rng=np.random.default_rng(os.getpid()))


def _pool_task(task):
    idx, direction = task
    rows = _profile_walk(_G["cost_func"], _G["param_opt"], _G["cost_opt"], idx, direction, _G["grids"][idx],
                         _G["calibr_setup"], _G["jac_spasity"], _G["cfg"], _G["rng"])
    return idx, direction, rows


def _run_walks(cost_func, param_opt, cost_opt, grids, tasks, calibr_setup, jac_spasity, cfg, n_jobs, on_done):
    """
    Parallelization:
      len(tasks) >= n_jobs : walks run in parallel, one (parameter, direction) walk per process
      len(tasks) <  n_jobs : walks run one after another, each fit evaluates its
                             finite-difference gradient in n_jobs processes (method='local')
    """
    n_jobs = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
    if n_jobs > 1 and len(tasks) >= n_jobs:
        print(f"Running {len(tasks)} walks in {n_jobs} parallel processes", flush=True)
        with mp.Pool(processes=n_jobs, initializer=_pool_init,
                     initargs=(cost_func, param_opt, cost_opt, grids, calibr_setup, jac_spasity, cfg)) as pool:
            for idx, direction, rows in pool.imap_unordered(_pool_task, tasks):
                on_done(idx, direction, rows)
        return
    rng = np.random.default_rng(0)
    use_grad_pool = n_jobs > 1 and len(tasks) > 0 and cfg["method"] == "local"
    if use_grad_pool:
        print(f"Running {len(tasks)} walks one after another with a parallel gradient in {n_jobs} processes",
              flush=True)
    with CostEvaluator(_TimedCost(cost_func, cfg["solve_timeout"]), calibr_setup, jac_spasity,
                       n_jobs=n_jobs if use_grad_pool else 1) as ev:
        cfg_serial = dict(cfg, evaluator=ev if use_grad_pool else None)
        for idx, direction in tasks:
            rows = _profile_walk(cost_func, param_opt, cost_opt, idx, direction, grids[idx],
                                 calibr_setup, jac_spasity, cfg_serial, rng)
            on_done(idx, direction, rows)


def _make_cfg(param_bnds, fixed_indices, method, confidence_level, scale, per_point_workers, n_restarts,
              jitter_frac, stop_factor, anchor_start, minimize_options, solve_timeout, verbose,
              param_names=None, progress=None):
    return dict(free_idx=free_param_indices(param_bnds, fixed_indices), method=method,
                confidence_level=confidence_level, scale=float(scale), per_point_workers=per_point_workers,
                n_restarts=n_restarts, jitter_frac=jitter_frac, stop_factor=stop_factor,
                anchor_start=anchor_start, minimize_options=minimize_options,
                solve_timeout=solve_timeout, verbose=verbose, param_names=param_names, progress=progress)


def _n_points_side(grid, p_hat, direction):
    return int(np.sum(grid > p_hat)) if direction > 0 else int(np.sum(grid < p_hat))


# ----------------------------------------------------------------------
# Single-parameter profile
# ----------------------------------------------------------------------

def profile_likelihood_for_param(
    cost_func,
    param_opt,
    param_index,
    calibr_setup,
    span=0.3,
    n_points=15,
    method="local",
    jac_spasity=None,
    n_jobs=1,
    per_point_workers=1,
    n_restarts=1,
    jitter_frac=0.05,
    fixed_indices=(),
    stop_factor=None,
    anchor_start=True,
    minimize_options=None,
    solve_timeout=30,
    verbose=True,
):
    """
    Scan one parameter around its estimated value, re-optimizing all other
    free parameters at each grid point (same signature and return value as
    pool_paper; extras after jitter_frac are optional).
    Returns (grid, profile_cost, profile_params); points not computed because
    of stop_factor have cost NaN.
    """
    param_opt = np.asarray(param_opt, dtype=float)
    grid = _build_grid(param_opt, param_index, calibr_setup["param_bnds"], span, n_points)
    cost_opt = _safe_cost(cost_func, param_opt, calibr_setup, jac_spasity, solve_timeout)
    progress = _Progress(len(grid) - 1)
    cfg = _make_cfg(calibr_setup["param_bnds"], fixed_indices, method, 0.95, 1.0, per_point_workers,
                    n_restarts, jitter_frac, stop_factor, anchor_start, minimize_options, solve_timeout, verbose,
                    progress=progress)
    print(f"Profiling param[{param_index}]: {len(grid) - 1} points, cost_opt = {cost_opt:.8g}", flush=True)
    results = {float(param_opt[param_index]): (cost_opt, param_opt.copy())}

    def on_done(idx, direction, rows):
        for r in rows:
            results[r["param_value"]] = (r["cost"], r["x"])

    _run_walks(cost_func, param_opt, cost_opt, {param_index: grid}, [(param_index, -1), (param_index, +1)],
               calibr_setup, jac_spasity, cfg, n_jobs, on_done)
    nan_p = np.full_like(param_opt, np.nan)
    profile_cost = np.array([results.get(float(g), (np.nan, None))[0] for g in grid])
    profile_params = np.array([results[float(g)][1] if float(g) in results else nan_p for g in grid])
    return grid, profile_cost, profile_params


# ----------------------------------------------------------------------
# Confidence intervals, scale
# ----------------------------------------------------------------------

def confidence_interval_from_profile(grid, profile_cost, cost_opt, confidence_level=0.95, dof=1, scale=1.0,
                                     interpolate=False):
    """
    threshold = chi2.ppf(confidence_level, dof) * scale
    CI = smallest/largest grid point with profile_cost - cost_opt <= threshold (same as pool_paper).
    interpolate=True (extra option): the bounds are instead the linearly interpolated
    threshold crossings between the last point below and the first point above the
    threshold, which does not shrink to single grid points on coarse grids.
    Use identifiability_from_profile to see whether the threshold is crossed on both sides.
    """
    threshold = chi2.ppf(confidence_level, dof) * scale
    grid = np.asarray(grid, dtype=float)
    d = np.asarray(profile_cost, dtype=float) - cost_opt
    below = np.where(d <= threshold)[0]
    if len(below) == 0:
        return None, None
    i_lo, i_hi = below[0], below[-1]
    lo, hi = grid[i_lo], grid[i_hi]
    if interpolate:
        if i_lo > 0 and np.isfinite(d[i_lo - 1]) and d[i_lo - 1] > d[i_lo]:
            lo = grid[i_lo] + (threshold - d[i_lo]) * (grid[i_lo - 1] - grid[i_lo]) / (d[i_lo - 1] - d[i_lo])
        if i_hi < len(grid) - 1 and np.isfinite(d[i_hi + 1]) and d[i_hi + 1] > d[i_hi]:
            hi = grid[i_hi] + (threshold - d[i_hi]) * (grid[i_hi + 1] - grid[i_hi]) / (d[i_hi + 1] - d[i_hi])
    return lo, hi


def identifiability_from_profile(grid, profile_cost, cost_opt, p_hat, confidence_level=0.95, dof=1, scale=1.0,
                                 bounds=None, flat_tol=0.05):
    """
    Classify one profile:
      'identifiable'                    threshold exceeded on both sides of the estimate
      'practically non-identifiable'    on an open side the profile reaches the parameter bound
                                        without exceeding the threshold
      'non-identifiable (flat)'         as above and max Delta < flat_tol * threshold
                                        (hint at structural non-identifiability)
      'undetermined (extend span)'      an open side ends inside the bounds: the grid is too
                                        narrow to decide -> increase span
    bounds : (lower, upper) of the parameter; if None, open sides count as reaching the bound.
    Returns (status, lower_closed, upper_closed).
    """
    threshold = chi2.ppf(confidence_level, dof) * scale
    d = np.asarray(profile_cost, dtype=float) - cost_opt
    grid = np.asarray(grid, dtype=float)
    lower_closed = bool(np.any(d[grid < p_hat] > threshold))
    upper_closed = bool(np.any(d[grid > p_hat] > threshold))
    if lower_closed and upper_closed:
        return "identifiable", True, True
    if bounds is not None:
        lo_b, hi_b = bounds
        tol = 1e-9 * max(1., abs(hi_b - lo_b))
        lower_at_bound = grid.min() <= lo_b + tol
        upper_at_bound = grid.max() >= hi_b - tol
        if (not lower_closed and not lower_at_bound) or (not upper_closed and not upper_at_bound):
            return "undetermined (extend span)", lower_closed, upper_closed
    if np.nanmax(d) < flat_tol * threshold:
        return "non-identifiable (flat)", lower_closed, upper_closed
    return "practically non-identifiable", lower_closed, upper_closed


def count_data_points(cost_func, param, calibr_setup, jac_spasity=None):
    """
    Number of residual terms feeding into cost_func(), counted by temporarily
    swapping in a counting aggregation_func. Unlike pool_paper (where only the
    first entry of J_vect is data and the rest is regularization), every entry
    of J_vect is data in fusion_model's cost functions ([ngs, maldi, mibi]),
    so all are counted (NaN = missing data excluded).
    """
    counting_setup = dict(calibr_setup)
    counts = {}

    def _counting_aggregation(J_vect):
        counts["n_data"] = int(sum(np.count_nonzero(np.isfinite(np.asarray(J, dtype=float))) for J in J_vect))
        return np.nanmean([np.nanmean(J) for J in J_vect])

    counting_setup["aggregation_func"] = _counting_aggregation
    cost_func(np.asarray(param, dtype=float), counting_setup, jac_spasity)
    return counts["n_data"]


def estimate_profile_scale(cost_func, param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=None):
    """
    Scale converting cost differences into Delta(-2 log L). Returns (scale, n_data, sigma_hat2).
    * aggregation_func is GaussianNeg2LogLik: the cost already is -2 log L -> scale = 1.
    * otherwise pool_paper's estimate for a mean squared residual under i.i.d. Gaussian noise:
      sigma_hat2 = cost_opt, scale = sigma_hat2 / n_data.
    """
    n_data = count_data_points(cost_func, param_opt, calibr_setup, jac_spasity)
    if isinstance(calibr_setup.get("aggregation_func"), GaussianNeg2LogLik):
        return 1.0, n_data, None
    if n_data <= 0:
        raise ValueError("n_data <= 0: no data points found in cost_func().")
    sigma_hat2 = cost_opt
    scale = sigma_hat2 / n_data
    return scale, n_data, sigma_hat2


# ----------------------------------------------------------------------
# Full run across all free parameters
# ----------------------------------------------------------------------

def run_profile_likelihood_all(
    cost_func,
    param_opt,
    calibr_setup,
    jac_spasity=None,
    span=0.3,
    n_points=15,
    method="local",
    confidence_level=0.95,
    scale="auto",
    n_jobs=1,
    per_point_workers=1,
    n_restarts=1,
    jitter_frac=0.05,
    out_csv="profile_likelihood_results.csv",
    param_names=None,
    plot=True,
    plot_path="profile_likelihood.png",
    ncols=4,
    # ---- optional extras (not in pool_paper) ----
    profile_indices=None,
    fixed_indices=(),
    refine=False,
    stop_factor=None,
    anchor_start=True,
    minimize_options=None,
    solve_timeout=30,
    resume=True,
    true_params=None,
    verbose=True,
):
    """
    Profile every free parameter (same arguments and return value as pool_paper):

    cost_func         : cost_func(param, calibr_setup, jac_spasity) -> float, e.g. fm.pest.cost_withS
    param_opt         : full estimated parameter vector (layout of calibr_setup['param_bnds'])
    span, n_points    : grid of n_points values in [p - span*|p|, p + span*|p|] (clipped to the bounds)
    method            : 'local' (L-BFGS-B, warm-started walks) or 'global' (differential_evolution)
    confidence_level  : pointwise threshold chi2.ppf(confidence_level, 1) * scale
    scale             : 'auto' (see estimate_profile_scale) or a float
    n_jobs            : processes. As many or more walks ((parameter, direction) pairs) than n_jobs:
                        one walk per process. Fewer walks, and the refit (refine=True): the
                        finite-difference gradient of each fit is evaluated in n_jobs processes.
    per_point_workers : differential_evolution workers per grid point (method='global' only)
    n_restarts        : starts per grid point incl. the warm start (others: random jitters)
    jitter_frac       : jitter size as a fraction of each parameter's bound width
    out_csv           : CSV with columns param_index, param_value, cost, <one column per parameter>
    param_names       : names for every entry of param_opt (CSV columns and plot titles)
    plot, plot_path, ncols : grid plot of all profiles

    Optional extras:
    profile_indices   : profile only these indices (default: all free)
    fixed_indices     : hold these at param_opt (neither profiled nor re-optimized), e.g. x0
    refine            : first polish param_opt locally with the profiling objective
    stop_factor       : stop a walk once Delta > stop_factor * threshold (None: whole grid, as pool_paper)
    anchor_start      : also start each point from param_opt (pool_paper's restart 0)
    minimize_options  : L-BFGS-B options (default maxiter=300, ftol=1e-10, gtol=1e-8)
    solve_timeout     : seconds per cost evaluation before it counts as failed
    resume            : skip walks already stored in <out_csv>_partial.csv (same cost_opt)
    true_params       : true parameter vector (in-silico data), marked in the plot
    verbose           : 0/False: only summaries; 1/True: one line per profile point (value, cost,
                        Delta, iterations, time, overall progress with ETA) and the refit's cost every
                        10 iterations; 2: additionally every 10 L-BFGS-B iterations inside each point

    Returns (df, ci_results) with ci_results = {param_index: (lo, hi)} as in pool_paper.
    Also writes <out_csv>_summary.json (cost_opt, scale, CIs, identifiability status).
    """
    param_opt = np.asarray(param_opt, dtype=float)
    bnds = calibr_setup["param_bnds"]
    if len(bnds) != len(param_opt):
        raise ValueError(f"len(param_opt)={len(param_opt)} != len(param_bnds)={len(bnds)}")
    names = list(param_names) if param_names is not None else [f"p{j}" for j in range(len(param_opt))]
    lo_all = np.array([b[0] for b in bnds]); hi_all = np.array([b[1] for b in bnds])
    param_opt = np.clip(param_opt, lo_all, hi_all)

    free_idx = free_param_indices(bnds, fixed_indices)
    if profile_indices is None:
        profile_indices = list(free_idx)
    skipped = [i for i in profile_indices if i not in free_idx]
    if skipped:
        print(f"Skipping fixed parameters (nothing to profile): {skipped}")
    profile_indices = [int(i) for i in profile_indices if i in free_idx]

    t_start = time.time()
    if refine:
        param_opt, cost_opt = refine_optimum(cost_func, param_opt, calibr_setup, jac_spasity,
                                             fixed_indices=fixed_indices, minimize_options=minimize_options,
                                             solve_timeout=solve_timeout, n_jobs=n_jobs,
                                             print_every=10 if verbose else 0)
    else:
        cost_opt = _safe_cost(cost_func, param_opt, calibr_setup, jac_spasity, solve_timeout)

    if scale == "auto":
        scale, n_data, sigma_hat2 = estimate_profile_scale(cost_func, param_opt, calibr_setup, cost_opt,
                                                           n_free_params=len(free_idx), jac_spasity=jac_spasity)
        print(f"Auto-estimated scale: n_data={n_data}, sigma_hat^2={sigma_hat2}, scale={scale:.6g}")
    scale = float(scale)
    threshold = chi2.ppf(confidence_level, 1) * scale

    grids = {idx: _build_grid(param_opt, idx, bnds, span, n_points) for idx in profile_indices}
    cfg = _make_cfg(bnds, fixed_indices, method, confidence_level, scale, per_point_workers, n_restarts,
                    jitter_frac, stop_factor, anchor_start, minimize_options, solve_timeout, verbose,
                    param_names=names)

    if method == "global" and n_jobs > 1:
        print("WARNING: method='global' already parallelizes within each grid point via "
              "differential_evolution(workers=...). Keep n_jobs * per_point_workers <= number of cores.")

    # --- resume -------------------------------------------------------
    partial_csv = os.path.splitext(out_csv)[0] + "_partial.csv"
    done_rows, done_tasks = [], set()
    if resume and os.path.exists(partial_csv):
        dfp = pd.read_csv(partial_csv)
        if len(dfp) and np.isclose(dfp["cost_opt"].iloc[0], cost_opt, rtol=1e-9, atol=1e-12):
            done_rows = dfp.to_dict("records")
            done_tasks = set(zip(dfp["param_index"].astype(int), dfp["direction"].astype(int)))
            print(f"Resuming: {len(done_tasks)} walks already in {partial_csv}")
        else:
            print(f"{partial_csv} belongs to a different optimum -- starting fresh.")
            os.remove(partial_csv)

    tasks = [(i, d) for i in profile_indices for d in (-1, +1) if (i, d) not in done_tasks]
    n_eval = sum(len(grids[i]) - 1 for i in profile_indices)
    n_todo = sum(_n_points_side(grids[i], param_opt[i], d) for i, d in tasks)
    cfg["progress"] = _Progress(n_todo)
    print(f"Profiling {len(profile_indices)} free parameters, {n_eval} total (parameter, value) evaluations "
          f"({len(tasks)} walks / {n_todo} points to compute, cost_opt={cost_opt:.8g}, "
          f"threshold={threshold:.6g}).", flush=True)
    if stop_factor is not None:
        print(f"  (walks stop once Delta > {stop_factor} x threshold, so fewer points may be needed; "
              f"skipped points count as done in the progress)", flush=True)

    def _records(rows):
        recs = []
        for r in rows:
            rec = {"param_index": r["param_index"], "direction": r["direction"], "param_value": r["param_value"],
                   "cost": r["cost"], "success": r["success"], "nit": r.get("nit"), "time_s": r.get("time_s"),
                   "cost_opt": cost_opt}
            for j, pv in enumerate(r["x"]):
                rec[names[j]] = pv
            recs.append(rec)
        return recs

    new_records = []
    state = {"done": 0}

    def on_done(idx, direction, rows):
        recs = _records(rows)
        if recs:
            pd.DataFrame(recs).to_csv(partial_csv, mode="a", index=False, header=not os.path.exists(partial_csv))
        new_records.extend(recs)
        state["done"] += 1
        d_max = max(((r["cost"] - cost_opt) / scale for r in rows), default=float("nan"))
        crossed = d_max > chi2.ppf(confidence_level, 1)
        print(f"[{time.strftime('%H:%M:%S')}] walk {state['done']}/{len(tasks)} finished: param[{idx}] {names[idx]} "
              f"({'+' if direction > 0 else '-'}), {len(rows)} points, max Delta = {d_max:.4g} "
              f"({'threshold crossed' if crossed else 'threshold NOT crossed'}), "
              f"elapsed {_fmt_time(time.time() - t_start)}", flush=True)

    _run_walks(cost_func, param_opt, cost_opt, grids, tasks, calibr_setup, jac_spasity, cfg, n_jobs, on_done)

    # --- assemble (pool_paper CSV format) -----------------------------
    opt_records = _records([{"param_index": i, "direction": 0, "param_value": float(param_opt[i]),
                             "cost": cost_opt, "success": True, "x": param_opt} for i in profile_indices])
    df_all = pd.DataFrame(done_rows + new_records + opt_records)
    df_all = df_all[df_all["param_index"].isin(profile_indices)]
    df = df_all[["param_index", "param_value", "cost"] + names].sort_values(
        ["param_index", "param_value"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved profile likelihood results to {out_csv}")

    min_delta = (df["cost"] - cost_opt).min()
    if min_delta < -1e-3 * threshold:
        best = df.loc[df["cost"].idxmin()]
        print(f"WARNING: a profile point has a lower cost than the optimum (Delta={min_delta / scale:.4g} at "
              f"param[{int(best['param_index'])}]={best['param_value']:.6g}); the estimate is not a minimum. "
              f"Consider refine=True or restarting from that point (full vector in the CSV).")

    ci_results, ci_interp, status = {}, {}, {}
    for idx in profile_indices:
        sub = df[df["param_index"] == idx]
        g, pc = sub["param_value"].to_numpy(), sub["cost"].to_numpy()
        ci_results[idx] = confidence_interval_from_profile(g, pc, cost_opt, confidence_level=confidence_level,
                                                           scale=scale)
        status[idx] = identifiability_from_profile(g, pc, cost_opt, param_opt[idx],
                                                   confidence_level=confidence_level, scale=scale,
                                                   bounds=bnds[idx])
        ci_interp[idx] = confidence_interval_from_profile(g, pc, cost_opt, confidence_level=confidence_level,
                                                          scale=scale, interpolate=True)

    print("\nApprox. confidence intervals (interpolated threshold crossings; "
          "* = threshold not exceeded on that side within the grid):")
    for idx in profile_indices:
        lo_ci, hi_ci = ci_interp[idx]
        st, lc, uc = status[idx]
        tv = "" if true_params is None else f"  true={true_params[idx]:.6g}"
        fmt = lambda v: "None" if v is None else f"{v:.6g}"  # noqa: E731
        print(f"  param[{idx}] {names[idx]} = {param_opt[idx]:.6g}  CI ~= ({fmt(lo_ci)}{'' if lc else '*'}, "
              f"{fmt(hi_ci)}{'' if uc else '*'})  {st}{tv}")

    summary_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"cost_opt": cost_opt, "scale": scale, "confidence_level": confidence_level,
                   "param_opt": param_opt.tolist(), "param_names": names,
                   "ci": {str(i): {"name": names[i], "estimate": float(param_opt[i]),
                                   "lower": None if ci_results[i][0] is None else float(ci_results[i][0]),
                                   "upper": None if ci_results[i][1] is None else float(ci_results[i][1]),
                                   "lower_interp": None if ci_interp[i][0] is None else float(ci_interp[i][0]),
                                   "upper_interp": None if ci_interp[i][1] is None else float(ci_interp[i][1]),
                                   "status": status[i][0], "lower_closed": status[i][1],
                                   "upper_closed": status[i][2]} for i in profile_indices}},
                  f, indent=2)
    print(f"Saved summary (cost_opt, scale, CIs, identifiability) to {summary_path}")
    print(f"Profile likelihood finished in {_fmt_time(time.time() - t_start)}", flush=True)

    if plot:
        plot_profile_likelihood(df, param_opt, cost_opt, ci_results=ci_results, confidence_level=confidence_level,
                                scale=scale, param_names=names, save_path=plot_path, ncols=ncols,
                                true_params=true_params)
    return df, ci_results


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _pct():
    return r"\%" if plt.rcParams["text.usetex"] else "%"


def _draw_profile(ax, x, y, p_hat, threshold_y, ci, true_val=None, confidence_level=0.95):
    ax.plot(x, y, "o-", color="#4E89B1", lw=1.5, ms=4)
    ax.axvline(p_hat, color="#D06062", ls="--", lw=1.2, label="estimate")
    if true_val is not None:
        ax.axvline(true_val, color="#386641", ls="-.", lw=1.2, label="true value")
    y_lo, y_hi = np.nanmin(y), np.nanmax(y)
    pad = 0.1 * max(y_hi - y_lo, 1e-8)
    ax.set_ylim(y_lo - pad, max(y_hi, threshold_y) + pad)
    ax.axhline(threshold_y, color="gray", ls=":", lw=1.2, label=f"{int(confidence_level*100)}{_pct()} threshold")
    if ci is not None and ci[0] is not None and ci[1] is not None:
        ax.axvspan(ci[0], ci[1], color="#4E89B1", alpha=0.12)


def plot_profile_likelihood(
    df,
    param_opt,
    cost_opt,
    ci_results=None,
    confidence_level=0.95,
    scale=1.0,
    param_names=None,
    save_path=None,
    ncols=4,
    true_params=None,
):
    """
    Grid of subplots, one per profiled parameter (same signature as pool_paper,
    plus optional true_params). The y-axis is Delta(-2 log L) = (cost - cost_opt) / scale.
    """
    free_idx = sorted(df["param_index"].unique())
    n = len(free_idx)
    ncols = min(ncols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))
    threshold = chi2.ppf(confidence_level, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, idx in zip(axes_flat, free_idx):
        sub = df[df["param_index"] == idx].sort_values("param_value")
        x = sub["param_value"].to_numpy()
        y = (sub["cost"].to_numpy() - cost_opt) / scale
        ci = ci_results.get(idx) if ci_results is not None else \
            confidence_interval_from_profile(x, sub["cost"].to_numpy(), cost_opt, confidence_level, scale=scale)
        _draw_profile(ax, x, y, param_opt[idx], threshold, ci,
                      None if true_params is None else true_params[idx], confidence_level)
        name = None
        if param_names is not None:
            name = param_names.get(idx) if isinstance(param_names, dict) else (
                param_names[idx] if idx < len(param_names) else None)
        ax.set_title(name if name else f"param[{idx}]", fontsize=12)
        ax.set_xlabel("parameter value", fontsize=10)
        ax.set_ylabel(r"$\Delta(-2\log L)$", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def _read_csv_and_summary(csv_path, scale):
    """CSV in pool_paper format + scale/cost_opt from <csv>_summary.json if present."""
    df = pd.read_csv(csv_path)
    names = [c for c in df.columns if c not in ("param_index", "param_value", "cost")]
    summ_path = os.path.splitext(csv_path)[0] + "_summary.json"
    summ = None
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            summ = json.load(f)
    cost_opt = summ["cost_opt"] if summ else df["cost"].min()
    if scale == "auto" or scale is None:
        if summ:
            scale = summ["scale"]
        else:
            raise ValueError("scale not provided and no <csv>_summary.json found: pass the scale "
                             "printed as 'Auto-estimated scale: ... scale=X' during the run.")
    if summ:
        param_opt = np.array(summ["param_opt"])
    else:  # estimate = grid point with the lowest cost of each profile
        param_opt = np.full(len(names), np.nan)
        for idx, sub in df.groupby("param_index"):
            param_opt[idx] = sub.loc[sub["cost"].idxmin(), "param_value"]
    return df, names, float(scale), float(cost_opt), param_opt


def plot_profile_likelihood_from_csv(
    csv_path,
    confidence_level=0.95,
    scale="auto",
    save_path=None,
    ncols=4,
    true_params=None,
):
    """
    Regenerate the grid plot from a CSV written by run_profile_likelihood_all
    (same signature as pool_paper's plot_profile_from_csv.py). scale='auto'
    reads the scale from <csv>_summary.json.
    """
    df, names, scale, cost_opt, param_opt = _read_csv_and_summary(csv_path, scale)
    ci = {idx: confidence_interval_from_profile(sub["param_value"].to_numpy(), sub["cost"].to_numpy(), cost_opt,
                                                confidence_level, scale=scale)
          for idx, sub in df.sort_values("param_value").groupby("param_index")}
    return plot_profile_likelihood(df, param_opt, cost_opt, ci_results=ci, confidence_level=confidence_level,
                                   scale=scale, param_names=names, save_path=save_path, ncols=ncols,
                                   true_params=true_params)


def plot_profile_likelihood_individual(
    csv_path,
    confidence_level=0.95,
    scale="auto",
    out_dir=".",
    file_prefix="profile_",
    file_ext="png",
    figsize=(5.5, 4.0),
    true_params=None,
):
    """
    One figure per parameter (as pool_paper's plot_profile_likelihood_individual,
    without the case-study-specific panel letters). Returns {param_index: file path}.
    """
    import re
    os.makedirs(out_dir, exist_ok=True)
    df, names, scale, cost_opt, param_opt = _read_csv_and_summary(csv_path, scale)
    threshold = chi2.ppf(confidence_level, 1)

    def _sanitize(name):
        s = re.sub(r"[\$\\{}\^]", "", name)
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    saved = {}
    for idx, sub in df.sort_values("param_value").groupby("param_index"):
        x = sub["param_value"].to_numpy()
        y = (sub["cost"].to_numpy() - cost_opt) / scale
        ci = confidence_interval_from_profile(x, sub["cost"].to_numpy(), cost_opt, confidence_level, scale=scale)
        fig, ax = plt.subplots(figsize=figsize)
        _draw_profile(ax, x, y, param_opt[idx], threshold, ci,
                      None if true_params is None else true_params[idx], confidence_level)
        ax.set_xlabel(names[idx], fontsize=16)
        ax.set_ylabel(r"$\Delta(-2\log L)$")
        fig.tight_layout()
        fpath = os.path.join(out_dir, f"{file_prefix}{idx:02d}_{_sanitize(names[idx])}.{file_ext}")
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved[idx] = fpath
        print(f"Saved {fpath}")
    return saved

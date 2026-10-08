"""
Profile likelihood with local minimization (fusion_model / model_paper)
=======================================================================

Same idea as ``pool_paper_casestudy/fusion_core/likelihood.py`` -- fix one
parameter on a grid, re-optimize all other free parameters with L-BFGS-B, and
compare the resulting cost with a chi2 threshold -- adapted to the
``fusion_model`` package and its cost convention

    cost_func(param, calibr_setup, jac_spasity) -> float

(e.g. ``fusion_model.parameter_estimation.cost_withS``).

Differences to the pool_paper engine
------------------------------------
* **Warm-started outward walks.** Each profile starts at the optimum and walks
  outwards (left and right separately). Every grid point is re-optimized
  starting from the solution of its neighbour, which is the standard way to
  compute profiles with a *local* optimizer (Raue et al. 2009): neighbouring
  optima are close, so L-BFGS-B converges quickly and stays on the same
  branch of the profile. Optional extra starts (from the optimum itself and
  random jitters) guard against getting stuck.
* **Adaptive step size + early stopping.** The step grows while the profile is
  flat and shrinks when it rises quickly; a walk stops once Delta exceeds
  ``stop_factor * threshold`` or the parameter bound is reached. This puts the
  points where they matter (around the threshold crossing) and saves a lot of
  ODE solves compared to a fixed grid.
* **Grid relative to the bounds**, not to the value (``lambda_1`` lives in
  [-7, -2], i.e. it is negative, so a multiplicative +/-span is ill-defined).
* **A proper -2 log L objective** (``GaussianNeg2LogLik``) as an alternative
  to the calibration aggregation. ``cost_sum_and_geometric_mean`` (used for
  calibration in model_paper) is not proportional to -2 log L, so a chi2
  threshold on it is not meaningful. ``GaussianNeg2LogLik`` assumes i.i.d.
  Gaussian noise with an unknown variance per data type (NGS, MALDI, MiBi)
  and profiles these variances out analytically:

      -2 log L = sum_k n_k * log(SSR_k / n_k) + const

  so Delta(-2 log L) can be compared directly with chi2 quantiles (scale = 1).
  n_k is counted from the measured data (finite entries), not from the
  residual arrays, so it stays constant when residuals hit exactly zero.
* **Fixed parameters** (lower bound == upper bound, e.g. the diagonal of the
  inhibition matrix k_ii or predefined S entries) are never optimized or
  profiled, and ``fixed_indices`` lets you freeze further parameters (e.g. the
  initial values x0) to speed things up.
* **Identifiability classification** per parameter (identifiable /
  practically non-identifiable / flat = structurally non-identifiable),
  confidence intervals from linear interpolation of the threshold crossing,
  and a CSV that stores Delta directly, so plots can be regenerated without
  knowing the scale.
* **Resumable**: finished (parameter, direction) walks are appended to a
  ``*_partial.csv`` file and skipped on restart with ``resume=True``.

Parallelization: one task = one (parameter, direction) walk; tasks run in a
``multiprocessing.Pool`` with ``n_jobs`` workers. A walk is inherently
sequential (warm start), so up to 2 * n_profiled workers are useful.
"""

import os
import json
import signal
import threading
import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

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
            if not np.isfinite(ssr):
                return PENALTY
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
def _time_limit(seconds):
    """Hard wall-clock limit (SIGALRM, Unix, main thread only; no-op otherwise)."""
    usable = (seconds and seconds > 0 and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def _handler(signum, frame):
        raise _SolveTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(np.ceil(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _safe_cost(cost_func, x_full, calibr_setup, jac_spasity, solve_timeout):
    try:
        with _time_limit(solve_timeout):
            c = float(cost_func(x_full, calibr_setup, jac_spasity))
    except Exception:
        return PENALTY
    return c if np.isfinite(c) else PENALTY


def free_param_indices(param_bnds, fixed_indices=()):
    """Indices with lower bound < upper bound that are not in fixed_indices."""
    fixed = set(int(i) for i in (fixed_indices or ()))
    return [j for j, (lo, hi) in enumerate(param_bnds) if hi > lo and j not in fixed]


def _local_fit(cost_func, x_start, opt_idx, calibr_setup, jac_spasity, *,
               extra_starts=(), n_jitter=0, jitter_frac=0.05, rng=None,
               minimize_options=None, solve_timeout=60):
    """
    Minimize cost_func over the entries opt_idx of the full parameter vector,
    all other entries held at their values in x_start. Runs L-BFGS-B from
    x_start, from every vector in extra_starts, and from n_jitter random
    perturbations of x_start; returns (best_cost, best_full_vector, success).
    """
    x_start = np.asarray(x_start, dtype=float)
    opt_idx = np.asarray(opt_idx, dtype=int)
    if len(opt_idx) == 0:
        return _safe_cost(cost_func, x_start, calibr_setup, jac_spasity, solve_timeout), x_start.copy(), True

    bnds = [calibr_setup["param_bnds"][j] for j in opt_idx]
    lo = np.array([b[0] for b in bnds], dtype=float)
    hi = np.array([b[1] for b in bnds], dtype=float)
    opts = {"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6}
    opts.update(minimize_options or {})
    # finite-difference gradients cost one evaluation per parameter; scipy's default
    # maxfun=15000 would otherwise stop large fits long before maxiter
    opts.setdefault("maxfun", 2 * (len(opt_idx) + 1) * opts["maxiter"])
    rng = rng if rng is not None else np.random.default_rng()

    def f(z):
        x = x_start.copy()
        x[opt_idx] = z
        return _safe_cost(cost_func, x, calibr_setup, jac_spasity, solve_timeout)

    starts = [x_start[opt_idx]]
    for xs in extra_starts:
        starts.append(np.asarray(xs, dtype=float)[opt_idx])
    for _ in range(int(n_jitter)):
        z = x_start[opt_idx] + rng.normal(0., jitter_frac, size=len(opt_idx)) * (hi - lo)
        starts.append(z)

    best_c, best_z, best_ok = np.inf, starts[0], False
    for z0 in starts:
        z0 = np.clip(z0, lo, hi)
        res = minimize(f, z0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=opts)
        if res.fun < best_c:
            best_c, best_z, best_ok = float(res.fun), res.x, bool(res.success)

    x_best = x_start.copy()
    x_best[opt_idx] = best_z
    return best_c, x_best, best_ok


def refine_optimum(cost_func, param_opt, calibr_setup, jac_spasity=None, fixed_indices=(),
                   n_jitter=0, jitter_frac=0.02, seed=0, minimize_options=None, solve_timeout=60):
    """
    Local L-BFGS-B polish of the calibration result with the *same* objective
    that is used for profiling. Needed because (i) the global calibration
    (differential_evolution, polish=False) does not end exactly at a local
    minimum and (ii) the profiling objective may differ from the calibration
    one (GaussianNeg2LogLik vs. cost_sum_and_geometric_mean). A profile must
    be referenced to the minimum of the profiled objective, otherwise
    Delta(-2 log L) becomes negative near the estimate.
    """
    param_opt = np.asarray(param_opt, dtype=float)
    opt_idx = free_param_indices(calibr_setup["param_bnds"], fixed_indices)
    c0 = _safe_cost(cost_func, param_opt, calibr_setup, jac_spasity, solve_timeout)
    c, x, ok = _local_fit(cost_func, param_opt, opt_idx, calibr_setup, jac_spasity,
                          n_jitter=n_jitter, jitter_frac=jitter_frac,
                          rng=np.random.default_rng(seed),
                          minimize_options=minimize_options, solve_timeout=solve_timeout)
    print(f"refine_optimum: cost {c0:.8g} -> {c:.8g} (success={ok}, {len(opt_idx)} free parameters)", flush=True)
    return x, c


# ----------------------------------------------------------------------
# One outward walk (one parameter, one direction)
# ----------------------------------------------------------------------

def _profile_walk(cost_func, param_opt, cost_opt, idx, direction, calibr_setup, jac_spasity, cfg, rng):
    """
    Walk parameter idx from its optimum towards its lower (direction=-1) or
    upper (direction=+1) bound, re-optimizing all other free parameters at
    every point, warm-started from the previous point.
    Returns a list of dicts (one per point).
    """
    lo, hi = calibr_setup["param_bnds"][idx]
    width = hi - lo
    threshold = cfg["threshold"]
    scale = cfg["scale"]
    opt_idx = [j for j in cfg["free_idx"] if j != idx]

    step = cfg["init_step_frac"] * width
    step_min = cfg["min_step_frac"] * width
    step_max = cfg["max_step_frac"] * width
    stop_delta = None if cfg["stop_factor"] is None else cfg["stop_factor"] * threshold

    x_prev = np.array(param_opt, dtype=float)
    val = float(param_opt[idx])
    delta_prev = 0.
    rows = []
    for k in range(cfg["max_points"]):
        bound = hi if direction > 0 else lo
        val_new = val + direction * step
        at_bound = (direction > 0 and val_new >= hi) or (direction < 0 and val_new <= lo)
        if at_bound:
            val_new = bound
        if np.isclose(val_new, val):
            break
        val = val_new

        x_start = x_prev.copy()
        x_start[idx] = val
        extra = []
        if cfg["anchor_start"]:
            x_anchor = np.array(param_opt, dtype=float)
            x_anchor[idx] = val
            extra.append(x_anchor)
        c, x_new, ok = _local_fit(cost_func, x_start, opt_idx, calibr_setup, jac_spasity,
                                  extra_starts=extra, n_jitter=cfg["n_jitter"],
                                  jitter_frac=cfg["jitter_frac"], rng=rng,
                                  minimize_options=cfg["minimize_options"],
                                  solve_timeout=cfg["solve_timeout"])
        delta = (c - cost_opt) / scale
        rows.append({"param_index": idx, "direction": direction, "step": k + 1,
                     "param_value": val, "cost": c, "delta": delta, "success": ok,
                     "x": x_new})
        if cfg["verbose"]:
            print(f"  p[{idx}] {'+' if direction > 0 else '-'} {k+1:2d}: value={val:.6g}  "
                  f"cost={c:.8g}  delta={delta:.4g}", flush=True)
        x_prev = x_new

        if at_bound:
            break
        if stop_delta is not None and delta > stop_delta:
            break
        # adapt the step to the local slope of the profile
        inc = abs(delta - delta_prev)
        if inc < 0.1 * threshold:
            step = min(step * 1.5, step_max)
        elif inc > 0.5 * threshold:
            step = max(step / 2., step_min)
        delta_prev = delta
    return rows


# --- multiprocessing plumbing (globals set once per worker) -----------
_G = {}


def _pool_init(cost_func, param_opt, cost_opt, calibr_setup, jac_spasity, cfg):
    _G.update(cost_func=cost_func, param_opt=param_opt, cost_opt=cost_opt,
              calibr_setup=calibr_setup, jac_spasity=jac_spasity, cfg=cfg,
              rng=np.random.default_rng(os.getpid()))


def _pool_task(task):
    idx, direction = task
    rows = _profile_walk(_G["cost_func"], _G["param_opt"], _G["cost_opt"], idx, direction,
                         _G["calibr_setup"], _G["jac_spasity"], _G["cfg"], _G["rng"])
    return idx, direction, rows


# ----------------------------------------------------------------------
# Confidence intervals and identifiability
# ----------------------------------------------------------------------

def _crossing(side_vals, side_deltas, p_hat, threshold):
    """First threshold crossing walking outward; None if never crossed."""
    v_prev, d_prev = p_hat, 0.
    for v, d in zip(side_vals, side_deltas):
        if d > threshold:
            if d == d_prev:
                return v
            return v_prev + (threshold - d_prev) * (v - v_prev) / (d - d_prev)
        v_prev, d_prev = v, d
    return None


def confidence_intervals_from_profiles(df, param_opt, param_bnds, threshold, flat_tol=None):
    """
    df : profile DataFrame (param_index, direction, param_value, delta).
    Returns {idx: dict(lower, upper, lower_open, upper_open, max_delta, status)}.

    status:
      'identifiable'                    both sides cross the threshold
      'practically non-identifiable'    at least one side never crosses (open CI)
      'non-identifiable (flat)'         profile essentially flat (max Delta < flat_tol)
                                        -> hint at structural non-identifiability
    """
    flat_tol = 0.05 * threshold if flat_tol is None else flat_tol
    out = {}
    for idx in sorted(df["param_index"].unique()):
        sub = df[df["param_index"] == idx]
        p_hat = float(param_opt[idx])
        res = {}
        for direction, key in ((-1, "lower"), (+1, "upper")):
            s = sub[sub["direction"] == direction].sort_values("step")
            v = _crossing(s["param_value"].to_numpy(), s["delta"].to_numpy(), p_hat, threshold)
            res[key] = v
            res[f"{key}_open"] = v is None
            if v is None:
                # report how far the walk got (bound, or last point evaluated)
                res[key] = float(s["param_value"].iloc[-1]) if len(s) else p_hat
        res["max_delta"] = float(np.nanmax(sub["delta"])) if len(sub) else 0.
        if res["lower_open"] or res["upper_open"]:
            res["status"] = "non-identifiable (flat)" if res["max_delta"] < flat_tol else "practically non-identifiable"
        else:
            res["status"] = "identifiable"
        out[int(idx)] = res
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def run_profile_likelihood(
    cost_func,
    param_opt,
    calibr_setup,
    jac_spasity=None,
    profile_indices=None,
    fixed_indices=(),
    refine=True,
    confidence_level=0.95,
    scale=1.0,
    init_step_frac=0.01,
    min_step_frac=1e-3,
    max_step_frac=0.1,
    max_points=30,
    stop_factor=1.5,
    anchor_start=True,
    n_jitter=0,
    jitter_frac=0.05,
    minimize_options=None,
    solve_timeout=60,
    n_jobs=1,
    param_names=None,
    out_csv="profile_likelihood_results.csv",
    resume=True,
    plot=True,
    plot_path="profile_likelihood.png",
    true_params=None,
    verbose=True,
):
    """
    Profile likelihood with local (L-BFGS-B) re-optimization.

    cost_func         : cost_func(param, calibr_setup, jac_spasity) -> float, e.g. fm.pest.cost_withS.
                        For a statistically meaningful chi2 threshold the cost should be -2 log L
                        (set calibr_setup['aggregation_func'] = make_gaussian_neg2loglik(data_array)
                        and keep scale=1). For any other cost, pass `scale` so that
                        (cost - cost_opt) / scale ~ Delta(-2 log L).
    param_opt         : full estimated parameter vector (same layout as calibr_setup['param_bnds']).
    profile_indices   : indices to profile (default: all free, non-fixed indices).
    fixed_indices     : indices held at param_opt (neither profiled nor re-optimized),
                        e.g. the initial values x0 to speed things up.
    refine            : first polish param_opt locally with the same objective (recommended).
    confidence_level  : pointwise threshold chi2.ppf(confidence_level, 1).
    init/min/max_step_frac : step sizes as fractions of the bound width of the profiled parameter.
    max_points        : max number of points per direction.
    stop_factor       : stop a walk once Delta > stop_factor * threshold (None: walk to the bound).
    anchor_start      : additionally start each inner fit from param_opt (with the profiled value
                        replaced), not only from the neighbouring point.
    n_jitter          : extra randomly-perturbed starts per point (jitter_frac * bound width).
    minimize_options  : passed to scipy L-BFGS-B (defaults maxiter=200, ftol=1e-10, gtol=1e-6).
    solve_timeout     : seconds per cost evaluation before it is treated as failed (PENALTY).
    n_jobs            : worker processes; one task = one (parameter, direction) walk.
    param_names       : list of names for every entry of param_opt (used in CSV/plots).
    out_csv           : results; finished walks are also appended to <out_csv>_partial.csv.
    resume            : skip walks already present in the partial CSV (same cost_opt required).
    true_params       : optional true parameter vector (in-silico data) to mark in the plot.

    Returns (df, ci, param_ref, cost_ref) where df holds all profile points
    (incl. the optimum), ci the confidence intervals/identifiability status per
    parameter, and param_ref/cost_ref the reference optimum.
    """
    param_opt = np.asarray(param_opt, dtype=float)
    bnds = calibr_setup["param_bnds"]
    n_par = len(param_opt)
    if len(bnds) != n_par:
        raise ValueError(f"len(param_opt)={n_par} != len(param_bnds)={len(bnds)}")
    names = list(param_names) if param_names is not None else [f"p{j}" for j in range(n_par)]

    # keep param_opt inside the bounds (saved results may sit marginally outside)
    lo_all = np.array([b[0] for b in bnds]); hi_all = np.array([b[1] for b in bnds])
    param_opt = np.clip(param_opt, lo_all, hi_all)

    free_idx = free_param_indices(bnds, fixed_indices)
    if profile_indices is None:
        profile_indices = list(free_idx)
    skipped = [i for i in profile_indices if i not in free_idx]
    if skipped:
        print(f"Skipping fixed parameters (no profile possible): {skipped}")
    profile_indices = [int(i) for i in profile_indices if i in free_idx]

    if refine:
        param_opt, cost_opt = refine_optimum(cost_func, param_opt, calibr_setup, jac_spasity,
                                             fixed_indices=fixed_indices,
                                             minimize_options=minimize_options,
                                             solve_timeout=solve_timeout)
    else:
        cost_opt = _safe_cost(cost_func, param_opt, calibr_setup, jac_spasity, solve_timeout)

    threshold = chi2.ppf(confidence_level, 1)
    cfg = dict(free_idx=free_idx, threshold=threshold, scale=float(scale),
               init_step_frac=init_step_frac, min_step_frac=min_step_frac,
               max_step_frac=max_step_frac, max_points=max_points, stop_factor=stop_factor,
               anchor_start=anchor_start, n_jitter=n_jitter, jitter_frac=jitter_frac,
               minimize_options=minimize_options, solve_timeout=solve_timeout,
               verbose=verbose)

    # --- resume -------------------------------------------------------
    partial_csv = os.path.splitext(out_csv)[0] + "_partial.csv"
    done_rows, done_tasks = [], set()
    if resume and os.path.exists(partial_csv):
        dfp = pd.read_csv(partial_csv)
        if "cost_opt" in dfp and np.allclose(dfp["cost_opt"].iloc[0], cost_opt, rtol=1e-8, atol=1e-10):
            done_rows = dfp.to_dict("records")
            done_tasks = set(zip(dfp["param_index"].astype(int), dfp["direction"].astype(int)))
            print(f"Resuming: {len(done_tasks)} walks already in {partial_csv}")
        else:
            print(f"{partial_csv} belongs to a different optimum -- starting fresh.")
            os.remove(partial_csv)

    tasks = [(i, d) for i in profile_indices for d in (-1, +1) if (i, d) not in done_tasks]
    print(f"Profiling {len(profile_indices)} parameters ({len(tasks)} walks to run, "
          f"{len(free_idx)} free parameters, threshold={threshold:.3f}, cost_opt={cost_opt:.8g})", flush=True)

    def _rows_to_records(rows):
        recs = []
        for r in rows:
            rec = {k: v for k, v in r.items() if k != "x"}
            rec["param_name"] = names[r["param_index"]]
            rec["cost_opt"] = cost_opt
            for j, pv in enumerate(r["x"]):
                rec[names[j]] = pv
            recs.append(rec)
        return recs

    def _append_partial(recs):
        if not recs:
            return
        pd.DataFrame(recs).to_csv(partial_csv, mode="a", index=False,
                                  header=not os.path.exists(partial_csv))

    new_records = []
    n_done = 0
    if n_jobs and n_jobs > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(n_jobs, len(tasks)), initializer=_pool_init,
                     initargs=(cost_func, param_opt, cost_opt, calibr_setup, jac_spasity, cfg)) as pool:
            for idx, direction, rows in pool.imap_unordered(_pool_task, tasks):
                recs = _rows_to_records(rows)
                _append_partial(recs)
                new_records += recs
                n_done += 1
                last = rows[-1] if rows else None
                print(f"[{n_done}/{len(tasks)}] {names[idx]} {'+' if direction > 0 else '-'}: "
                      f"{len(rows)} points" + (f", last delta={last['delta']:.3g} at {last['param_value']:.4g}" if last else ""),
                      flush=True)
    else:
        rng = np.random.default_rng(0)
        for idx, direction in tasks:
            rows = _profile_walk(cost_func, param_opt, cost_opt, idx, direction,
                                 calibr_setup, jac_spasity, cfg, rng)
            recs = _rows_to_records(rows)
            _append_partial(recs)
            new_records += recs
            n_done += 1
            print(f"[{n_done}/{len(tasks)}] {names[idx]} {'+' if direction > 0 else '-'} done", flush=True)

    # optimum rows (direction 0) so every profile contains its minimum
    opt_records = _rows_to_records([{"param_index": i, "direction": 0, "step": 0,
                                     "param_value": param_opt[i], "cost": cost_opt, "delta": 0.,
                                     "success": True, "x": param_opt} for i in profile_indices])
    df = pd.DataFrame(done_rows + new_records + opt_records)
    df = df[df["param_index"].isin(profile_indices)]

    # a profile point below the optimum means the optimum was not the minimum
    min_delta = df["delta"].min()
    if min_delta < -1e-3 * threshold:
        best = df.loc[df["delta"].idxmin()]
        print(f"WARNING: a profile point has lower cost than the optimum (delta={min_delta:.4g} at "
              f"{best['param_name']}={best['param_value']:.6g}). The estimate is not the global/local "
              f"minimum; consider restarting from that point (its full parameter vector is in the CSV).")

    cols = ["param_index", "param_name", "direction", "step", "param_value", "cost", "delta", "success", "cost_opt"]
    df = df[cols + names].sort_values(["param_index", "param_value"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} profile points to {out_csv}")

    ci = confidence_intervals_from_profiles(df, param_opt, bnds, threshold)
    _print_ci_table(ci, param_opt, names, true_params)
    summary_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"cost_opt": cost_opt, "threshold": threshold, "confidence_level": confidence_level,
                   "scale": scale, "param_opt": param_opt.tolist(), "param_names": names,
                   "ci": {names[i]: {**v, "index": i, "estimate": float(param_opt[i])} for i, v in ci.items()}},
                  f, indent=2)
    print(f"Saved summary to {summary_path}")

    if plot:
        plot_profile_likelihood(df, param_opt, ci, threshold, param_names=names,
                                true_params=true_params, save_path=plot_path,
                                confidence_level=confidence_level)
    return df, ci, param_opt, cost_opt


def _print_ci_table(ci, param_opt, names, true_params=None):
    print(f"\n{'parameter':<28}{'estimate':>12}{'lower':>12}{'upper':>12}  status")
    for idx, r in ci.items():
        lo = f"{r['lower']:.4g}" + ("*" if r["lower_open"] else "")
        hi = f"{r['upper']:.4g}" + ("*" if r["upper_open"] else "")
        tv = ""
        if true_params is not None:
            inside = (r["lower"] <= true_params[idx] <= r["upper"]) and not (r["lower_open"] or r["upper_open"])
            tv = f"   true={true_params[idx]:.4g}" + (" (in CI)" if inside else "")
        print(f"{names[idx]:<28}{param_opt[idx]:>12.4g}{lo:>12}{hi:>12}  {r['status']}{tv}")
    print("(* = threshold not crossed on that side; value is the last point reached)\n")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_profile_likelihood(df, param_opt, ci, threshold, param_names=None, true_params=None,
                            save_path=None, ncols=4, confidence_level=0.95, indices=None, usetex=None):
    """Delta(-2 log L) vs. parameter value for every profiled parameter.
    usetex: None keeps the current matplotlib rc setting, True/False overrides it."""
    rc = {} if usetex is None else {"text.usetex": bool(usetex)}
    with plt.rc_context(rc):
        return _plot_profile_likelihood(df, param_opt, ci, threshold, param_names, true_params,
                                        save_path, ncols, confidence_level, indices)


def _plot_profile_likelihood(df, param_opt, ci, threshold, param_names, true_params,
                             save_path, ncols, confidence_level, indices):
    pct = r"\%" if plt.rcParams["text.usetex"] else "%"
    idxs = sorted(df["param_index"].unique()) if indices is None else list(indices)
    n = len(idxs)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, idx in zip(axes, idxs):
        sub = df[df["param_index"] == idx].sort_values("param_value")
        x, y = sub["param_value"].to_numpy(), sub["delta"].to_numpy()
        ax.plot(x, y, "o-", color="#4E89B1", lw=1.4, ms=3)
        ax.axvline(param_opt[idx], color="#D06062", ls="--", lw=1.1, label="estimate")
        if true_params is not None:
            ax.axvline(true_params[idx], color="#386641", ls="-.", lw=1.1, label="true value")
        ax.axhline(threshold, color="gray", ls=":", lw=1.2, label=f"{int(confidence_level*100)}{pct} threshold")
        r = ci.get(int(idx)) if ci else None
        if r is not None:
            ax.axvspan(r["lower"], r["upper"], color="#4E89B1", alpha=0.12)
            if r["status"] != "identifiable":
                ax.text(0.03, 0.95, r["status"].replace("practically ", "pract. "), transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, color="#99582A")
        ymax = max(np.nanmax(y), threshold) * 1.15
        ax.set_ylim(min(-0.05 * ymax, np.nanmin(y)), ymax)
        name = param_names[idx] if param_names is not None else f"p[{idx}]"
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel(r"$\Delta(-2\log L)$", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def plot_profile_likelihood_from_csv(csv_path, true_params=None, save_path=None, ncols=4, indices=None, usetex=None):
    """Regenerate the plot from the CSV + summary JSON written by run_profile_likelihood."""
    df = pd.read_csv(csv_path)
    with open(os.path.splitext(csv_path)[0] + "_summary.json") as f:
        summ = json.load(f)
    param_opt = np.array(summ["param_opt"])
    ci = {v["index"]: v for v in summ["ci"].values()}
    return plot_profile_likelihood(df, param_opt, ci, summ["threshold"], param_names=summ["param_names"],
                                   true_params=true_params, save_path=save_path, ncols=ncols,
                                   confidence_level=summ["confidence_level"], indices=indices, usetex=usetex)

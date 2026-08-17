"""
Profile likelihood for pool_paper_casestudy (MDFM repo)
=========================================================

Adapted to the `cost(param, calibr_setup, jac_spasity)` signature used in
`pool_paper_casestudy/do_local_optim.py`, following the same fix-one-parameter /
re-optimize-the-rest pattern already used in
`fusion_model/parameter_estimation/likelihood_functions.py::calculate_profile_likelihood`
(that generic version uses a different `ll_func(param, dfs, model, P_matrix, s_x)`
signature and won't run as-is on this case study).

Includes:
  - multiprocessing across grid points (one worker process per (parameter, value) pair)
  - plotting of the resulting profile-likelihood curves with the chi2-based
    threshold and confidence interval

CAVEAT ON THE CHI2 THRESHOLD
-----------------------------
`cost_arithmetic_mean` (the aggregation function used here) returns a *mean
squared residual*, not `-2*logL`. The chi2-based confidence interval below is
only statistically exact if `cost` is proportional to `-2*logL` (e.g.
`n * MSE / sigma^2` under i.i.d. Gaussian noise). Treat `confidence_interval_from_profile`
as a convenience utility -- calibrate `scale` (or pass your own threshold) if you
need a rigorous interval. This mirrors what `likelihood_functions.py` already
does elsewhere in the repo, so it is consistent with how the rest of the codebase
reports these intervals, but it is an approximation, not a guarantee.

CAVEAT ON MULTIPROCESSING
---------------------------
`method="global"` already parallelizes *within* each grid point via
`differential_evolution(..., workers=...)`. Also parallelizing *across* grid
points (n_jobs > 1) on top of that oversubscribes your CPU cores. If you use
`method="global"`, keep `n_jobs=1` (default) and control parallelism only
through `per_point_workers`, or explicitly divide your cores between the two
levels (n_jobs * per_point_workers <= number of cores).
`method="local"` (L-BFGS-B) is single-threaded per point, so `n_jobs > 1` is
the recommended way to parallelize it.

Drop this file into `pool_paper_casestudy/` (same level as `do_local_optim.py`)
so the relative imports resolve, or adjust the `sys.path` / import lines below.
"""

import os
import sys

sys.path.append(os.getcwd())
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

import fusion_model as fm
from pool_paper_casestudy.pool_model_functions import *
from pool_paper_casestudy.do_local_optim import cost  # your cost(param, calibr_setup, jac_spasity)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def free_param_indices(param_bnds):
    """Indices whose bounds are not fixed (lo == hi)."""
    return [j for j, (lo, hi) in enumerate(param_bnds) if hi > lo]


def _cost_fixed_param(free_params, fixed_index, fixed_value, calibr_setup, jac_spasity):
    """cost() with parameter `fixed_index` clamped to `fixed_value`, rest free."""
    full_params = np.insert(np.asarray(free_params, dtype=float), fixed_index, fixed_value)
    return cost(full_params, calibr_setup, jac_spasity)


def _build_grid(param_opt, param_index, param_bnds, span, n_points):
    p_opt_val = param_opt[param_index]
    lo, hi = param_bnds[param_index]
    if hi <= lo:
        raise ValueError(f"param_index {param_index} is fixed in param_bnds ({lo}, {hi}); nothing to profile.")
    if p_opt_val == 0:
        grid = np.linspace(lo, hi, n_points)
    else:
        grid_lo = max(lo, p_opt_val * (1 - span))
        grid_hi = min(hi, p_opt_val * (1 + span))
        grid = np.linspace(grid_lo, grid_hi, n_points)
    # make sure the optimum itself is included so profile_cost has a true minimum
    return np.sort(np.unique(np.concatenate([grid, [p_opt_val]])))


def _optimize_one_point(param_index, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers):
    """Re-optimize all free parameters except `param_index` (fixed at `val`)."""
    free_idx = [j for j in range(len(param_opt)) if j != param_index]
    free_bnds = [calibr_setup["param_bnds"][j] for j in free_idx]
    x0_free = param_opt[free_idx]

    if method == "local":
        res = minimize(
            _cost_fixed_param,
            x0_free,
            args=(param_index, val, calibr_setup, jac_spasity),
            method="L-BFGS-B",
            bounds=free_bnds,
            options={"maxiter": 200},
        )
        best_free, best_cost = res.x, res.fun
    elif method == "global":
        setup_local = dict(calibr_setup)
        setup_local["workers"] = per_point_workers
        res = fm.pest.optimization_func(
            _cost_fixed_param,
            free_bnds,
            args=(param_index, val, setup_local, jac_spasity),
            workers=per_point_workers,
        )
        best_free, best_cost = res.x, res.fun
    else:
        raise ValueError("method must be 'local' or 'global'")

    full_p = np.insert(best_free, param_index, val)
    return best_cost, full_p


# ----------------------------------------------------------------------
# Multiprocessing across grid points
# ----------------------------------------------------------------------
# Globals set once per worker process via the Pool initializer, so
# calibr_setup / param_opt aren't re-pickled for every single task.
_G_PARAM_OPT = None
_G_CALIBR_SETUP = None
_G_JAC = None
_G_METHOD = None
_G_PER_POINT_WORKERS = None


def _pool_init(param_opt, calibr_setup, jac_spasity, method, per_point_workers):
    global _G_PARAM_OPT, _G_CALIBR_SETUP, _G_JAC, _G_METHOD, _G_PER_POINT_WORKERS
    _G_PARAM_OPT = param_opt
    _G_CALIBR_SETUP = calibr_setup
    _G_JAC = jac_spasity
    _G_METHOD = method
    _G_PER_POINT_WORKERS = per_point_workers


def _pool_task(task):
    param_index, val = task
    best_cost, full_p = _optimize_one_point(
        param_index, val, _G_PARAM_OPT, _G_CALIBR_SETUP, _G_JAC, _G_METHOD, _G_PER_POINT_WORKERS
    )
    print(f"  param[{param_index}] = {val:.6g}  ->  cost = {best_cost:.6g}", flush=True)
    return param_index, val, best_cost, full_p


# ----------------------------------------------------------------------
# Single-parameter profile (kept for profiling just one parameter at a time)
# ----------------------------------------------------------------------

def profile_likelihood_for_param(
    param_opt,
    param_index,
    calibr_setup,
    span=0.3,
    n_points=15,
    method="local",
    jac_spasity=None,
    n_jobs=1,
    per_point_workers=1,
):
    """
    Scan one parameter around its estimated value, re-optimizing all other
    free parameters at each grid point, and record the resulting cost.

    param_opt          : full estimated parameter vector (x0's + ODE params), 1D array,
                          e.g. from Result_calibration.json / Result_calibration_local.json
    param_index        : index into param_opt / calibr_setup['param_bnds'] to profile
    span               : +/- fractional range around the optimum to scan (e.g. 0.3 = +/-30%),
                          clipped to the parameter's bounds
    n_points           : number of grid points
    method             : 'local'  -> scipy.optimize.minimize(L-BFGS-B) from the current best
                                      (fast, matches do_local_optim.py's local refinement)
                          'global' -> fm.pest.optimization_func (differential_evolution),
                                      slower but more robust against local minima
    n_jobs             : number of worker processes to run grid points in parallel.
                          n_jobs=1 -> serial (default). See CAVEAT ON MULTIPROCESSING above
                          before combining n_jobs > 1 with method='global'.
    per_point_workers  : workers passed to differential_evolution *within* each grid
                          point when method='global' (ignored for method='local').
    """
    param_opt = np.asarray(param_opt, dtype=float)
    grid = _build_grid(param_opt, param_index, calibr_setup["param_bnds"], span, n_points)
    tasks = [(param_index, val) for val in grid]

    results = {}
    if n_jobs and n_jobs > 1:
        with mp.Pool(
            processes=n_jobs,
            initializer=_pool_init,
            initargs=(param_opt, calibr_setup, jac_spasity, method, per_point_workers),
        ) as pool:
            for idx, val, best_cost, full_p in pool.imap_unordered(_pool_task, tasks):
                results[val] = (best_cost, full_p)
    else:
        for _, val in tasks:
            best_cost, full_p = _optimize_one_point(
                param_index, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers
            )
            print(f"  param[{param_index}] = {val:.6g}  ->  cost = {best_cost:.6g}")
            results[val] = (best_cost, full_p)

    profile_cost = np.array([results[val][0] for val in grid])
    profile_params = np.array([results[val][1] for val in grid])
    return grid, profile_cost, profile_params


def confidence_interval_from_profile(grid, profile_cost, cost_opt, confidence_level=0.95, dof=1, scale=1.0):
    """
    threshold = chi2.ppf(confidence_level, dof) * scale
    CI = smallest/largest grid point with profile_cost - cost_opt <= threshold

    See the CAVEAT at the top of this file: this is exact only if `cost` is
    proportional to -2*logL. `scale` lets you calibrate that proportionality
    (e.g. scale = sigma_hat**2 / n if you've estimated a residual variance).
    """
    threshold = chi2.ppf(confidence_level, dof) * scale
    below = np.where((profile_cost - cost_opt) <= threshold)[0]
    if len(below) == 0:
        return None, None
    return grid[below[0]], grid[below[-1]]


# ----------------------------------------------------------------------
# Full run across all free parameters, with a single shared process pool
# ----------------------------------------------------------------------

def run_profile_likelihood_all(
    param_opt,
    calibr_setup,
    jac_spasity=None,
    span=0.3,
    n_points=15,
    method="local",
    confidence_level=0.95,
    scale=1.0,
    n_jobs=1,
    per_point_workers=1,
    out_csv="profile_likelihood_results.csv",
    param_names=None,
    plot=True,
    plot_path="profile_likelihood.png",
    ncols=4,
):
    """
    Loop profile_likelihood_for_param over every *free* parameter in param_opt.
    All (parameter, grid-point) tasks across *all* parameters are flattened into
    one pool of `n_jobs` worker processes (more efficient than re-spawning a
    pool per parameter). Saves a tidy CSV and, by default, a plot.
    """
    param_opt = np.asarray(param_opt, dtype=float)
    cost_opt = cost(param_opt, calibr_setup, jac_spasity)
    free_idx = free_param_indices(calibr_setup["param_bnds"])

    grids = {idx: _build_grid(param_opt, idx, calibr_setup["param_bnds"], span, n_points) for idx in free_idx}
    all_tasks = [(idx, val) for idx in free_idx for val in grids[idx]]
    print(f"Profiling {len(free_idx)} free parameters, {len(all_tasks)} total (parameter, value) evaluations.")

    if method == "global" and n_jobs > 1:
        print(
            "WARNING: method='global' already parallelizes within each grid point via "
            "differential_evolution(workers=...). Combining this with n_jobs>1 oversubscribes "
            "your CPU cores -- consider n_jobs=1 with a larger per_point_workers, or method='local'."
        )

    results = {idx: {} for idx in free_idx}
    progress = _make_progress_printer(len(all_tasks), step_pct=10)
    if n_jobs and n_jobs > 1:
        with mp.Pool(
            processes=n_jobs,
            initializer=_pool_init,
            initargs=(param_opt, calibr_setup, jac_spasity, method, per_point_workers),
        ) as pool:
            for idx, val, best_cost, full_p in pool.imap_unordered(_pool_task, all_tasks):
                results[idx][val] = (best_cost, full_p)
                progress()
    else:
        for idx, val in all_tasks:
            best_cost, full_p = _optimize_one_point(
                idx, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers
            )
            print(f"  param[{idx}] = {val:.6g}  ->  cost = {best_cost:.6g}")
            results[idx][val] = (best_cost, full_p)
            progress()

    rows = []
    ci_results = {}
    for idx in free_idx:
        grid = grids[idx]
        profile_cost = np.array([results[idx][val][0] for val in grid])
        lo_ci, hi_ci = confidence_interval_from_profile(
            grid, profile_cost, cost_opt, confidence_level=confidence_level, scale=scale
        )
        ci_results[idx] = (lo_ci, hi_ci)
        for g, c in zip(grid, profile_cost):
            full_p = results[idx][g][1]  # full re-optimized parameter vector at this point
            row = {"param_index": idx, "param_value": g, "cost": c}
            for j, pv in enumerate(full_p):
                col_name = param_names[j] if param_names is not None and j < len(param_names) else f"p{j}"
                row[col_name] = pv
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved profile likelihood results to {out_csv}")
    print("\nApprox. confidence intervals (see CAVEAT in this file's docstring):")
    for idx, (lo_ci, hi_ci) in ci_results.items():
        print(f"  param[{idx}] = {param_opt[idx]:.6g}  CI ~= ({lo_ci}, {hi_ci})")

    if plot:
        plot_profile_likelihood(
            df, param_opt, cost_opt, ci_results=ci_results,
            confidence_level=confidence_level, scale=scale,
            param_names=param_names, save_path=plot_path, ncols=ncols,
        )

    return df, ci_results

def _make_progress_printer(total, step_pct=10):
    """Returns a callback that prints once every step_pct% of `total` completed tasks."""
    state = {"done": 0, "next_threshold": step_pct}

    def _report():
        state["done"] += 1
        pct = 100 * state["done"] / total
        if pct >= state["next_threshold"]:
            print(f"  progress: {state['done']}/{total} ({pct:.0f}%)", flush=True)
            while state["next_threshold"] <= pct:
                state["next_threshold"] += step_pct

    return _report

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

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
):
    """
    Grid of subplots, one per profiled parameter: cost vs. parameter value,
    with the optimum marked, the chi2 threshold line, and the (approximate)
    confidence interval shaded.

    df          : DataFrame with columns ['param_index', 'param_value', 'cost'],
                  as returned by run_profile_likelihood_all
    param_opt   : full estimated parameter vector
    cost_opt    : cost at param_opt (the profile minimum reference)
    ci_results  : dict {param_index: (lo, hi)} as returned by run_profile_likelihood_all
                  (optional -- recomputed per-parameter if not given)
    param_names : optional dict/list mapping param_index -> display name
    save_path   : if given, saves the figure to this path (png/pdf/...)
    ncols       : number of subplot columns
    """
    free_idx = sorted(df["param_index"].unique())
    n = len(free_idx)
    ncols = min(ncols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))

    threshold = chi2.ppf(confidence_level, 1) * scale

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, idx in enumerate(free_idx):
        ax = axes_flat[ax_i]
        sub = df[df["param_index"] == idx].sort_values("param_value")
        x = sub["param_value"].to_numpy()
        y = sub["cost"].to_numpy()

        ax.plot(x, y, "o-", color="#4E89B1", lw=1.5, ms=4)
        ax.axvline(param_opt[idx], color="#D06062", ls="--", lw=1.2, label="estimate")
        ax.axhline(cost_opt + threshold, color="gray", ls=":", lw=1.2, label=f"{int(confidence_level*100)}% threshold")

        if ci_results is not None and idx in ci_results:
            lo_ci, hi_ci = ci_results[idx]
        else:
            lo_ci, hi_ci = confidence_interval_from_profile(x, y, cost_opt, confidence_level, scale=scale)
        if lo_ci is not None and hi_ci is not None:
            ax.axvspan(lo_ci, hi_ci, color="#4E89B1", alpha=0.12)

        name = None
        if param_names is not None:
            name = param_names.get(idx) if isinstance(param_names, dict) else (
                param_names[idx] if idx < len(param_names) else None
            )
        ax.set_title(name if name else f"param[{idx}]", fontsize=11)
        ax.set_xlabel("parameter value", fontsize=9)
        ax.set_ylabel("cost", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax_j in range(n, len(axes_flat)):
        axes_flat[ax_j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Example usage -- adapt to how you actually built calibr_setup
    # and obtained param_opt in your run of do_local_optim.py /
    # case_study_poolpaper2.py.
    # --------------------------------------------------------------
    path2 = "pool_paper_casestudy/out/all_together_final/"
    n_cl = 4

    result = fm.output.read_from_json("Result_calibration_5exps_local.json", dir="pool_paper_casestudy/out/all_together_final/")
    param_opt = np.array(result["param_ode"])
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    data_array = extract_observables_from_df([dfs])
    x0_vals = param_opt[:n_cl*len(exps)]
    param_ode = param_opt[n_cl*len(exps):]
    model = ode_model_coculture3
    calibr_presetup = {
            "model": model,
            "output_path": path2,
            "n_cl": n_cl,
            "dfs": [dfs],
            "aggregation_func": fm.pest.cost_arithmetic_mean,
            "exps": exps,
            'data_array': data_array,
            'x0': x0_vals
    }
    param_ode_bnds = tuple(
            [(.2, 1.) for _ in range (3)] + # mu_opt
            [(1., 3.5), (6., 8.), (9., 14.),
             (1., 3.5), (6., 8.), (9., 14.),
             (1., 3.5), (6., 8.), (9., 14.)] +  # pH_min, pH_opt, pH_max
            [(0.5, 2.), (3000., 5000.), (0.3, 1.5)] + # omegaT_exp + ki_T_inhib + n  
            [(8., 9.), (8., 9.), (8., 9.)]  + # N_max_exp
            [(.1, 1.)] + # kappa_T
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA ls23K
            [(.1, 10)] + [(1., 100.)] +   # kappa_LA lsCTC494
            [(.1, 10)] + [(1., 100.)]     # kappa_LA lm
        )
    #param_ode_bnds = [(p, p) for p in param_ode]
    #param_ode_bnds[3*4:3*4+3] = [(0.5, 10.), (10., 10000.), (0.1, 3.)]
    #param_ode_bnds = tuple(param_ode_bnds)
    calibr_setup = calibr_presetup
    calibr_setup["param_bnds"] = param_ode_bnds

    ode_param_names = [
        "mu_ls23K_opt", "mu_lsCTC494_opt", "mu_lm_opt",
        "pH_ls23K_min", "pH_ls23K_opt", "pH_ls23K_max",
        "pH_lsCTC494_min", "pH_lsCTC494_opt", "pH_lsCTC494_max",
        "pH_lm_min", "pH_lm_opt", "pH_lm_max",
        "omegaT_lm", "k_T_inhib", "n",
        "N_ls23K_texp", "N_lsCTC494_texp", "N_lm_texp",
        "kappa_T_0",
        "kappa_LA_ls23K_exp", "kappa_LA_ls23K_2_exp",
        "kappa_LA_lsCTC494_exp", "kappa_LA_lsCTC494_2_exp",
        "kappa_LA_lm_exp", "kappa_LA_lm_2_exp",
    ]

    df, cis = run_profile_likelihood_all(
         param_ode, calibr_setup,
         span=0.9, n_points=12, method="local",
         n_jobs=20,              # parallelize across grid points (safe for method='local')
         per_point_workers=20,
         out_csv=path2+"profile_likelihood_results.csv",
         plot_path=path2+"profile_likelihood.png",
        param_names=ode_param_names,
    )
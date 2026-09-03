
import concurrent.futures
import signal
from contextlib import contextmanager

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _SolveTimeout(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Raises _SolveTimeout if the wrapped block runs longer than `seconds`.
    Unix-only (uses SIGALRM); must run in the main thread of a process --
    fine here since each multiprocessing.Pool worker runs its task in its
    own process's main thread."""
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _SolveTimeout(f"solve exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(np.ceil(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def free_param_indices(param_bnds):
    """Indices whose bounds are not fixed (lo == hi)."""
    return [j for j, (lo, hi) in enumerate(param_bnds) if hi > lo]


def _cost_fixed_param(free_params, fixed_index, fixed_value, calibr_setup, jac_spasity, solve_timeout=30):
    """cost() with parameter `fixed_index` clamped to `fixed_value`, rest free.
    Wraps the ODE solve in a hard wall-clock timeout (seconds) so a
    pathological parameter combination can't hang the whole sweep."""
    full_params = np.insert(np.asarray(free_params, dtype=float), fixed_index, fixed_value)
    try:
        with time_limit(solve_timeout):
            c = cost(full_params, calibr_setup, jac_spasity)
        if not np.isfinite(c):
            return 1e3
        return c
    except _SolveTimeout:
        return 1e3
    except Exception:
        return 1e3


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


def _optimize_one_point(
    param_index, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers,
    n_restarts=1, jitter_frac=0.05, rng=None,
):
    """
    Re-optimize all free parameters except `param_index` (fixed at `val`).

    n_restarts : number of inner optimizations to run from slightly different
                 starting points (only applies to method='local'); the best
                 (lowest-cost) result across restarts is kept. Restart 0 always
                 uses the exact warm start (param_opt); restarts 1..n-1 use a
                 random jitter around it. Guards against reporting a cost that's
                 only high because a single deterministic L-BFGS-B run got stuck.
    jitter_frac : relative size of the random perturbation applied to each free
                 parameter's starting value for restarts > 0 (clipped to bounds).
    rng         : np.random.Generator; a fresh default one is created if None
                 (each worker process should get its own via the pool initializer,
                 not share one across processes).
    """
    free_idx = [j for j in range(len(param_opt)) if j != param_index]
    free_bnds = [calibr_setup["param_bnds"][j] for j in free_idx]
    x0_free = param_opt[free_idx]

    if method == "local":
        if rng is None:
            rng = np.random.default_rng()

        best_free, best_cost = None, np.inf
        for r in range(max(1, n_restarts)):
            if r == 0:
                x0_r = x0_free
            else:
                lo_arr = np.array([b[0] for b in free_bnds])
                hi_arr = np.array([b[1] for b in free_bnds])
                span = hi_arr - lo_arr
                jitter = rng.normal(0.0, jitter_frac, size=x0_free.shape) * np.where(span > 0, span, 1.0)
                x0_r = np.clip(x0_free + jitter, lo_arr, hi_arr)

            res = minimize(
                _cost_fixed_param,
                x0_r,
                args=(param_index, val, calibr_setup, jac_spasity),
                method="L-BFGS-B",
                bounds=free_bnds,
                options={"maxiter": 200},
            )
            if res.fun < best_cost:
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
_G_N_RESTARTS = None
_G_JITTER_FRAC = None
_G_RNG = None


def _pool_init(param_opt, calibr_setup, jac_spasity, method, per_point_workers, n_restarts=1, jitter_frac=0.05):
    global _G_PARAM_OPT, _G_CALIBR_SETUP, _G_JAC, _G_METHOD, _G_PER_POINT_WORKERS
    global _G_N_RESTARTS, _G_JITTER_FRAC, _G_RNG
    _G_PARAM_OPT = param_opt
    _G_CALIBR_SETUP = calibr_setup
    _G_JAC = jac_spasity
    _G_METHOD = method
    _G_PER_POINT_WORKERS = per_point_workers
    _G_N_RESTARTS = n_restarts
    _G_JITTER_FRAC = jitter_frac
    # seed differently per worker (PID) so restarts aren't identical across processes
    _G_RNG = np.random.default_rng(os.getpid())


def _pool_task(task):
    param_index, val = task
    best_cost, full_p = _optimize_one_point(
        param_index, val, _G_PARAM_OPT, _G_CALIBR_SETUP, _G_JAC, _G_METHOD, _G_PER_POINT_WORKERS,
        n_restarts=_G_N_RESTARTS, jitter_frac=_G_JITTER_FRAC, rng=_G_RNG,
    )
    print(f"  param[{param_index}] = {val:.6g}  ->  cost = {best_cost:.6g}", flush=True)
    return param_index, val, best_cost, full_p


def _pool_task_with_timeout(task, timeout=60):
    param_index, val = task
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            _optimize_one_point, param_index, val, _G_PARAM_OPT, _G_CALIBR_SETUP,
            _G_JAC, _G_METHOD, _G_PER_POINT_WORKERS, _G_N_RESTARTS, _G_JITTER_FRAC, _G_RNG,
        )
        try:
            best_cost, full_p = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"  TIMEOUT at param[{param_index}] = {val:.6g} (>{timeout}s) -- skipping", flush=True)
            return param_index, val, np.nan, np.full_like(_G_PARAM_OPT, np.nan)
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
    n_restarts=1,
    jitter_frac=0.05,
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
            initargs=(param_opt, calibr_setup, jac_spasity, method, per_point_workers, n_restarts, jitter_frac),
        ) as pool:
            for idx, val, best_cost, full_p in pool.imap_unordered(_pool_task, tasks):
                results[val] = (best_cost, full_p)
    else:
        rng = np.random.default_rng()
        for _, val in tasks:
            best_cost, full_p = _optimize_one_point(
                param_index, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers,
                n_restarts=n_restarts, jitter_frac=jitter_frac, rng=rng,
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


def count_data_points(param, calibr_setup, jac_spasity=None):
    """
    Count the number of individual residual terms feeding into cost()'s `ll_x`
    block (the real data residuals), by temporarily swapping in a counting
    aggregation_func instead of cost_arithmetic_mean. Excludes the
    regularization term (min(0, param)**2) from the count -- that's a
    parameter-positivity penalty, not real data.
    """
    counting_setup = dict(calibr_setup)
    counts = {}

    def _counting_aggregation(J_vect):
        ll_x = np.asarray(J_vect[0])
        counts["n_data"] = int(np.size(ll_x[~np.isnan(ll_x)])) if ll_x.size else 0
        # still return a real number so cost() doesn't error downstream
        return np.nanmean([np.nanmean(Ji) for Ji in J_vect])

    counting_setup["aggregation_func"] = _counting_aggregation
    cost(param, counting_setup, jac_spasity)  # runs cost() only to trigger the count
    return counts["n_data"]


def estimate_profile_scale(param_opt, calibr_setup, cost_opt, n_free_params, jac_spasity=None):
    """
    Estimate the scale factor converting cost() (mean squared residual) into
    an approximate -2*logL scale for the chi2 threshold, assuming i.i.d.
    Gaussian residual noise:

        cost_profile - cost_opt <= chi2.ppf(confidence_level, 1) * scale
        sigma_hat^2 = cost_opt * n_data / (n_data - n_free_params)   (bias-corrected)
        scale = sigma_hat^2 / n_data

    Returns (scale, n_data, sigma_hat2). Raises if n_data <= n_free_params
    (model over-parameterized relative to the data actually used in cost()).
    """
    n_data = count_data_points(param_opt, calibr_setup, jac_spasity)
    dof_resid = n_data - n_free_params
    if dof_resid <= 0:
        raise ValueError(
            f"n_data ({n_data}) <= n_free_params ({n_free_params}); can't estimate "
            "residual variance -- model is over-parameterized relative to the data "
            "actually used in cost()."
        )
    sigma_hat2 = cost_opt * n_data / dof_resid
    scale = sigma_hat2 / n_data
    return scale, n_data, sigma_hat2

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

    if scale == "auto":
        scale, n_data, sigma_hat2 = estimate_profile_scale(
            param_opt, calibr_setup, cost_opt, n_free_params=len(free_idx), jac_spasity=jac_spasity
        )
        print(f"Auto-estimated scale: n_data={n_data}, sigma_hat^2={sigma_hat2:.6g}, scale={scale:.6g}")

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
            initargs=(param_opt, calibr_setup, jac_spasity, method, per_point_workers, n_restarts, jitter_frac),
        ) as pool:
            for idx, val, best_cost, full_p in pool.imap_unordered(_pool_task, all_tasks):
                results[idx][val] = (best_cost, full_p)
                progress()
    else:
        rng = np.random.default_rng()
        for idx, val in all_tasks:
            best_cost, full_p = _optimize_one_point(
                idx, val, param_opt, calibr_setup, jac_spasity, method, per_point_workers,
                n_restarts=n_restarts, jitter_frac=jitter_frac, rng=rng,
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

        # zoom to the data, not the threshold line
        y_lo, y_hi = np.min(y), np.max(y)
        pad = 0.1 * max(y_hi - y_lo, 1e-8)
        ax.set_ylim(y_lo - pad, y_hi + pad)

        thresh_y = cost_opt + threshold
        if thresh_y <= y_hi + pad:
            ax.axhline(thresh_y, color="gray", ls=":", lw=1.2, label=f"{int(confidence_level*100)}% threshold")
        else:
            ax.text(0.98, 0.95, "threshold off-scale", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="gray")

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
    path2 = "pool_paper_casestudy/out/wo_pH_new/"
    n_cl = 4

    result = fm.output.read_from_json("Result_calibration_5exps_local.json", dir=path2)
    param_opt = np.array(result["param_ode"])
    dfs = pd.read_pickle(path2+f'dataframe_poolpaper_all.pkl')
    exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
    data_array = extract_observables_from_df([dfs])
    x0_vals = param_opt[:n_cl*len(exps)]
    param_ode = param_opt[n_cl*len(exps):]
    model = ode_model_coculture_wopH
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

    #ode_param_names = [
    #    "mu_ls23K_opt", "mu_lsCTC494_opt", "mu_lm_opt",
    #    #"pH_ls23K_min", "pH_ls23K_opt", "pH_ls23K_max",
    #    #"pH_lsCTC494_min", "pH_lsCTC494_opt", "pH_lsCTC494_max",
    #    #"pH_lm_min", "pH_lm_opt", "pH_lm_max",
    #    "omegaT_lm", "k_T_inhib", "n",
    #    "N_ls23K_texp", "N_lsCTC494_texp", "N_lm_texp",
    #    "kappa_T_0",
    #    "kappa_LA_ls23K_exp", "kappa_LA_ls23K_2_exp",
    #    "kappa_LA_lsCTC494_exp", "kappa_LA_lsCTC494_2_exp",
    #    "kappa_LA_lm_exp", "kappa_LA_lm_2_exp",
    #]

    ode_param_names = [
        r"$\mu_{Ls23K}$", r"$\mu_{LsCTC494}$", r"$\mu_{Lm}$",
        r"$\omegaT_{Lm}$", r"$K_{inhib}$", r"$n$",
        r"$N^{Ls23K}_{texp}$", r"$N^{LsCTC494}_{texp}$", r"$N^{Lm}_{texp}$",
        r"$\kappa_{T} \cdot 10^{-5}$",
        r"$\kappa_{LA}^{Ls23K} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{Ls23K} \cdot 10^{-9}$",
        r"$\kappa_{LA}^{LsCTC494} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{LsCTC494} \cdot 10^{-9}$",
        r"$\kappa_{LA}^{Lm} \cdot 10^{-9}$", r"$\kappa_{LA/G}^{Lm} \cdot 10^{-9}$",
    ]

    df, cis = run_profile_likelihood_all(
         param_ode, calibr_setup,
         span=0.9, n_points=9, method="local",
         n_jobs=1,              # parallelize across grid points (safe for method='local')
         per_point_workers=20,
         out_csv=path2+"profile_likelihood_results.csv",
         plot_path=path2+"profile_likelihood.png",
        param_names=ode_param_names,
)
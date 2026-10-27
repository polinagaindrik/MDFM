#!/usr/bin/env python3

import os
import multiprocessing as mp
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from .. import model as mdl
from fusion_model.tools.dataframe_functions import extract_observables_from_df


optimization_history = []
#output_file = 'model_paper/out/optimization_history1.csv'
#output_file2 = 'model_paper/out/optimization_history2.csv'
#output_file_predict = 'model_paper/out/optimization_history_predict.csv'
output_file = 'out/optimization_history1.csv'
output_file2 = 'out/optimization_history2.csv'
output_file_predict = 'out/optimization_history_predict.csv'
output_file_local = 'out/optimization_history_local.csv'


def calculate_model_params(cost_func, calibr_setup, local_polish=False, local_options=None, maxiter=1000):
    """
    Global optimization (differential_evolution) of the model parameters.

    local_polish  : if True, the best point found by the global search is refined by a
                    local L-BFGS-B minimization (see local_optimization). The global search
                    stops after few generations and without scipy's own polishing, so its
                    result is usually close to, but not at, a local minimum.
    local_options : dict of keyword arguments for local_optimization
                    (e.g. {'maxiter': 1000, 'n_restarts': 3}).
    maxiter       : max. number of differential evolution generations.
    Returns (param_opt, cost_opt); with local_polish the refined values.
    """
    jac_spasity = mdl.jacobian_sparsity(np.shape(calibr_setup['dfs'][1])[0])
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"p{i},"
        f.write(output+"cost\n")
    data_array = extract_observables_from_df(calibr_setup['dfs'])
    calibr_setup['data_array'] = data_array
    optim_output = optimization_func(cost_func, calibr_setup['param_bnds'], args=(calibr_setup, jac_spasity),
                                     workers=calibr_setup['workers'], maxiter=maxiter)
    param_opt, cost_opt = np.array(optim_output.x), optim_output.fun
    if local_polish:
        param_opt, cost_opt = local_optimization(cost_func, param_opt, calibr_setup, jac_spasity=jac_spasity,
                                                 **(local_options or {}))
    return param_opt, cost_opt


# Local refinement of a (global) optimization result
# ----------------------------------------------------------------------
# Parallel cost evaluation / finite-difference gradient for L-BFGS-B
# ----------------------------------------------------------------------
_EVAL = {}


def die_with_parent():
    """
    Worker initializer: the worker gets SIGTERM as soon as its parent process dies
    (Linux prctl PR_SET_PDEATHSIG). Without this, killing the main process (kill <PID>,
    or a crash) leaves the pool workers running as orphans. No-op on other systems.
    """
    try:
        import ctypes
        import signal as _signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, _signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
        if os.getppid() == 1:  # parent already gone before prctl took effect
            os._exit(1)
    except Exception:
        pass


def jax_worker_init():
    """Initializer of 'spawn' worker processes for the JAX backend: die with the parent, one thread
    per process (the processes themselves run in parallel), no GPU memory preallocation."""
    die_with_parent()
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def _eval_init(cost_func, calibr_setup, jac_spasity):
    """Initialize a worker's shared objective and calibration arguments."""
    die_with_parent()
    _EVAL.update(cost_func=cost_func, calibr_setup=calibr_setup, jac_spasity=jac_spasity)


def _eval_one(x):
    """Evaluate one parameter vector using the worker's configured objective."""
    try:
        c = float(_EVAL['cost_func'](x, _EVAL['calibr_setup'], _EVAL['jac_spasity']))
    except Exception:
        return 1e10
    return c if np.isfinite(c) else 1e10


class CostEvaluator:
    """
    Evaluates cost_func(x, calibr_setup, jac_spasity) for a list of full parameter
    vectors, in a pool of n_jobs processes (n_jobs > 1) or serially. Use as a
    context manager so the pool is closed afterwards.
    """
    def __init__(self, cost_func, calibr_setup, jac_spasity=None, n_jobs=1):
        """Create a serial or process-pool evaluator for the supplied objective."""
        self.n_jobs = int(n_jobs) if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
        self.pool = None
        self.args = (cost_func, calibr_setup, jac_spasity)
        if self.n_jobs > 1:
            self.pool = mp.Pool(processes=self.n_jobs, initializer=_eval_init, initargs=self.args)
        else:
            _eval_init(*self.args)

    def __call__(self, xs):
        """Evaluate the objective for each full parameter vector in xs."""
        if self.pool is not None:
            return np.array(self.pool.map(_eval_one, xs, chunksize=1))
        return np.array([_eval_one(x) for x in xs])

    def close(self):
        """Shut down the worker pool and restore the serial evaluator state."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        _eval_init(*self.args)  # later calls still work (serially) instead of returning the penalty

    def __enter__(self):
        """Return this evaluator for use in a context manager."""
        return self

    def __exit__(self, exc_type, *exc):
        """Close the evaluator, terminating workers if the managed block failed."""
        if exc_type is not None and self.pool is not None:  # error / Ctrl+C: don't wait for running tasks
            self.pool.terminate()
            self.pool = None
        self.close()


def fd_fun_and_grad(evaluator, x_template, free, lo, hi, rel_step=None):
    """
    Returns fg(z) -> (cost, gradient) for minimize(..., jac=True), where z are the
    entries `free` of the full vector x_template. The gradient is a forward finite
    difference with scipy's default step (sqrt(eps) * max(1, |z|)); a step that would
    leave the bounds is taken backwards. All n_free + 1 cost evaluations of one call
    are done in one batch by `evaluator` (in parallel if it has a pool).
    """
    x_template = np.asarray(x_template, dtype=float)
    free = np.asarray(free, dtype=int)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    rel_step = np.sqrt(np.finfo(float).eps) if rel_step is None else rel_step

    def fg(z):
        """Evaluate cost and a finite-difference gradient for the free parameters."""
        z = np.asarray(z, dtype=float)
        x0 = x_template.copy()
        x0[free] = z
        h = rel_step * np.maximum(1., np.abs(z))
        h = np.where(z + h > hi, -h, h)
        h = (z + h) - z  # exactly representable step
        xs = [x0]
        for i, j in enumerate(free):
            x = x0.copy()
            x[j] = z[i] + h[i]
            xs.append(x)
        c = evaluator(xs)
        return float(c[0]), (c[1:] - c[0]) / h

    return fg


class _LocalObjective:
    """Cost as a function of the free parameters only (picklable for multiprocessing)."""
    def __init__(self, cost_func, param_init, free, calibr_setup, jac_spasity):
        """Store the full parameter template and the indices optimized locally."""
        self.cost_func, self.param_init, self.free = cost_func, param_init, free
        self.calibr_setup, self.jac_spasity = calibr_setup, jac_spasity

    def full(self, z):
        """Insert free-parameter values into a copy of the full parameter vector."""
        x = self.param_init.copy()
        x[self.free] = z
        return x

    def __call__(self, z):
        """Evaluate the objective for free-parameter values, returning a penalty on failure."""
        try:
            c = self.cost_func(self.full(z), self.calibr_setup, self.jac_spasity)
        except Exception:
            return 1e10
        return c if np.isfinite(c) else 1e10


def _local_run(task):
    """One L-BFGS-B run (one start of the multistart).
    grad_jobs > 1: the finite-difference gradient is evaluated in a pool of grad_jobs
    processes (only possible in the main process, i.e. when the starts run sequentially)."""
    r, objective, z0, bounds, options, print_every, grad_jobs = task
    history = []
    backend = options.get('backend', 'numpy')
    if options.get('method') == 'least_squares':
        from .least_squares_fit import least_squares_fit
        x_start = objective.full(z0)
        c, x, ok, njev = least_squares_fit(
            objective.cost_func, x_start, objective.free, objective.calibr_setup,
            residual_func=options.get('residual_func'), jac_spasity=objective.jac_spasity, n_jobs=grad_jobs,
            log_scale=options.get('log_scale', 'auto'), maxiter=options['maxiter'],
            max_outer=options.get('max_outer', 8), label=f'[start {r}]', print_every=print_every,
            history=history, backend=backend)
        print(f'  [start {r}] cost = {c:.10g}, Jacobians = {njev}, success = {ok} (least_squares)', flush=True)
        return r, np.array(x)[objective.free], float(c), bool(ok), history

    def callback(intermediate_result):
        """Record the current full parameter vector and cost during a local run."""
        history.append((objective.full(intermediate_result.x), float(intermediate_result.fun)))
        if print_every and len(history) % print_every == 0:
            print(f'  [start {r}] iteration {len(history)}: cost = {intermediate_result.fun:.10g}', flush=True)

    lbfgs_options = {k: v for k, v in options.items() if k != 'backend'}
    if backend == 'jax':  # exact gradient by automatic differentiation
        from .jax_backend import jax_cost_for
        jc = jax_cost_for(objective.calibr_setup)

        def fg(z):
            c, g = jc.value_and_grad(objective.full(z))
            return c, g[objective.free]
        res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=bounds, callback=callback, options=lbfgs_options)
    elif grad_jobs > 1:
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        with CostEvaluator(objective.cost_func, objective.calibr_setup, objective.jac_spasity,
                           n_jobs=grad_jobs) as ev:
            fg = fd_fun_and_grad(ev, objective.param_init, objective.free, lo, hi)
            res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
                           options=lbfgs_options)
    else:
        res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, callback=callback, options=lbfgs_options)
    print(f'  [start {r}] cost = {res.fun:.10g}, iterations = {res.nit}, success = {res.success} ({res.message})',
          flush=True)
    return r, np.array(res.x), float(res.fun), bool(res.success), history


def apply_fixed_values(calibr_setup, fixed_values, param=None, names=None, verbose=True):
    """
    Hold parameters at given values (e.g. non-identifiable ones set to 0): returns a copy of
    calibr_setup whose param_bnds are (value, value) for these indices, so that every optimizer
    and the profile likelihood treat them as fixed; the value may lie outside the normal bounds.
    fixed_values : {index: value}
    param        : full parameter vector; if given, a copy with the fixed values is returned too
    Returns calibr_setup (and the parameter vector if param is given).
    """
    fixed_values = {int(i): float(v) for i, v in (fixed_values or {}).items()}
    bnds = list(calibr_setup['param_bnds'])
    for i, v in fixed_values.items():
        if not 0 <= i < len(bnds):
            raise IndexError(f'fixed parameter index {i} outside 0..{len(bnds) - 1}')
        if verbose:
            old = bnds[i]
            name = names[i] if names is not None else f'param[{i}]'
            print(f'Fixed: [{i}] {name} = {v:g} (bounds were {old})', flush=True)
        bnds[i] = (v, v)
    setup = dict(calibr_setup, param_bnds=tuple(bnds))
    if param is None:
        return setup
    x = np.array(param, dtype=float)
    for i, v in fixed_values.items():
        x[i] = v
    return setup, x


def fixed_values_record(fixed_values, names=None):
    """{'index': {'name': ..., 'value': ...}} for the result JSON (see apply_fixed_values)."""
    return {str(int(i)): {'name': None if names is None else names[int(i)], 'value': float(v)}
            for i, v in (fixed_values or {}).items()}


def local_optimization(cost_func, param_init, calibr_setup, jac_spasity=None, maxiter=500, ftol=1e-12, gtol=1e-8,
                       n_restarts=1, jitter_frac=0.02, seed=0, history_file=output_file_local, print_every=10,
                       n_jobs=1, maxfun=None, method='L-BFGS-B', residual_func=None, log_scale='auto',
                       start_points=None, max_outer=8, fixed_values=None, backend='numpy'):
    """
    L-BFGS-B minimization of cost_func starting from param_init (e.g. the result of
    differential_evolution) to reach the exact local optimum.

    method      : 'L-BFGS-B' (default) or 'least_squares': trust-region least squares on the
                  residuals with reweighting for the aggregation, parameters scaled to [0, 1]
                  (log10 for wide positive ranges, log_scale='auto'), see least_squares_fit.py.
                  Much faster convergence for ill-conditioned problems. maxiter is then the max.
                  number of residual evaluations per least-squares solve, max_outer the max.
                  number of reweighting steps. residual_func: signed residuals belonging to
                  cost_func (default: known for fm.pest.cost_withS).
    start_points: additional full start vectors (e.g. Latin hypercube points, see
                  multistart_optimization); start 0 is param_init, then the jittered starts.
    backend     : 'numpy' (finite-difference derivatives, as before) or 'jax' (exact gradient / Jacobian by
                  automatic differentiation, jax_backend.py; much faster, ODE solved more accurately). With
                  'jax' the starts run in n_jobs separate processes ('spawn'), each single-threaded.
    fixed_values: {index: value}: these parameters are held at value (also outside their bounds,
                  e.g. 0 for a non-identifiable k_ij) and only the others are refined; the returned
                  vector contains the fixed values. See apply_fixed_values.

    Only free parameters (lower bound < upper bound) are optimized; fixed ones (e.g. the
    diagonal k_ii = 0 or predefined S entries) stay at their value.
    n_restarts > 1 adds starts from random perturbations of param_init (jitter_frac times the
    bound width); the best result is kept. Start 0 is always param_init itself.
    maxfun      : max. number of cost evaluations per start. The gradient is computed by finite
                  differences (one evaluation per free parameter), so scipy's default of 15000 stops
                  a 74-parameter fit after ~170 iterations. Default None: 2*(n_free+1)*maxiter, i.e.
                  maxiter is the effective limit.
    n_jobs      : number of processes (-1 = all CPU cores).
                  n_jobs <= n_restarts: the starts run in parallel, one start per process.
                  n_jobs >  n_restarts: the starts run one after another and each uses n_jobs
                  processes for its finite-difference gradient (n_free + 1 cost evaluations
                  per iteration), which keeps all cores busy even for a single start.
    The cost of every iteration of the best run is written to history_file (columns as in
    optimization_history1.csv); the history of every start is written to
    <history_file>_start<r>.csv when n_restarts > 1.
    Returns (param_opt, cost_opt). The local result is only accepted if it is not worse
    than param_init.
    """
    param_init = np.asarray(param_init, dtype=float)
    if backend == 'jax':  # same (more accurate) solver for start cost, fits and result
        from .cost_function import cost_withS, cost_direct
        from .jax_backend import cost_withS_jax, cost_direct_jax
        if cost_func is cost_withS:
            cost_func = cost_withS_jax
        elif cost_func is cost_direct:
            cost_func = cost_direct_jax
    if fixed_values:
        calibr_setup = apply_fixed_values(calibr_setup, fixed_values)
    bnds = calibr_setup['param_bnds']
    lo = np.array([b[0] for b in bnds], dtype=float)
    hi = np.array([b[1] for b in bnds], dtype=float)
    param_init = np.clip(param_init, lo, hi)
    free = np.where(hi > lo)[0]
    rng = np.random.default_rng(seed)
    objective = _LocalObjective(cost_func, param_init, free, calibr_setup, jac_spasity)

    cost_init = objective(param_init[free])
    print(f'Local optimization: start cost = {cost_init:.10g} ({len(free)} free parameters)', flush=True)

    starts = [param_init[free]]
    for _ in range(max(0, n_restarts - 1)):
        z = param_init[free] + rng.normal(0., jitter_frac, size=len(free)) * (hi[free] - lo[free])
        starts.append(np.clip(z, lo[free], hi[free]))
    for xs in (start_points or []):
        starts.append(np.clip(np.asarray(xs, dtype=float)[free], lo[free], hi[free]))

    bounds = list(zip(lo[free], hi[free]))
    if method == 'least_squares':
        options = {'method': 'least_squares', 'maxiter': maxiter, 'residual_func': residual_func,
                   'log_scale': log_scale, 'max_outer': max_outer, 'backend': backend}
        print(f'Local optimization method: least_squares (trust-region reflective, reweighted, '
              f'log_scale={log_scale})', flush=True)
    elif method == 'L-BFGS-B':
        if maxfun is None:
            maxfun = 2 * (len(free) + 1) * maxiter
        options = {'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol, 'backend': backend}
    else:
        raise ValueError("method must be 'L-BFGS-B' or 'least_squares'")
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    grad_jobs = n_jobs if n_jobs > len(starts) and backend != 'jax' else 1
    tasks = [(r, objective, z0, bounds, options, print_every, grad_jobs) for r, z0 in enumerate(starts)]

    n_proc = 1 if grad_jobs > 1 else min(n_jobs, len(tasks))
    if n_proc > 1:
        print(f'Running {len(tasks)} local optimizations in {n_proc} parallel processes', flush=True)
        if backend == 'jax':  # JAX does not survive fork: fresh single-threaded processes
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_proc, initializer=jax_worker_init) as pool:
                results = list(pool.imap_unordered(_local_run, tasks))
        else:
            with mp.Pool(processes=n_proc, initializer=die_with_parent) as pool:
                results = list(pool.imap_unordered(_local_run, tasks))
    else:
        if grad_jobs > 1:
            print(f'Running {len(tasks)} local optimization(s) one after another, each with a parallel '
                  f'gradient in {grad_jobs} processes', flush=True)
        results = [_local_run(t) for t in tasks]
    results.sort(key=lambda res: res[0])

    if history_file:
        os.makedirs(os.path.dirname(history_file) or '.', exist_ok=True)

        def _write_history(fname, history):
            """Write a local optimization run's parameter and cost history to CSV."""
            with open(fname, 'w') as fh:
                fh.write('iteration,' + ''.join(f'p{i},' for i in range(len(param_init))) + 'cost\n')
                for it, (x, c) in enumerate(history, 1):
                    fh.write(f'{it},' + ''.join(f'{p},' for p in x) + f'{c}\n')

        if len(results) > 1:
            base, ext = os.path.splitext(history_file)
            for r, _, _, _, history in results:
                _write_history(f'{base}_start{r}{ext}', history)

    if len(results) > 1:
        print('Local optimization summary (sorted by cost):', flush=True)
        c_min = min(res[2] for res in results)
        tol = 1e-6 * max(1., abs(c_min))
        for r, _, c, ok, _ in sorted(results, key=lambda res: res[2]):
            print(f'  start {r}: cost = {c:.10g}, success = {ok}{"  <- best" if c - c_min <= tol else ""}', flush=True)
        n_best = sum(res[2] - c_min <= tol for res in results)
        print(f'  {n_best} of {len(results)} starts reached the best cost '
              f'({"several starts agree" if n_best > 1 else "only one start: run more starts to be sure"})',
              flush=True)

    r_best, z_best, c_best, _, history_best = min(results, key=lambda res: res[2])
    if history_file:
        _write_history(history_file, history_best)

    if c_best > cost_init:
        print('Local optimization did not improve the cost; keeping the initial parameters.', flush=True)
        return param_init, cost_init
    print(f'Local optimization: cost {cost_init:.10g} -> {c_best:.10g} (best start: {r_best})', flush=True)
    return objective.full(z_best), c_best


def multistart_optimization(cost_func, calibr_setup, n_starts=20, seed=0, extra_starts=None, method='least_squares',
                            maxiter=200, n_jobs=1, history_file=output_file_local, log_scale='auto', **kwargs):
    """
    Global search by many local fits from Latin hypercube start points over the bounds
    (log10 for wide positive ranges, as in the least-squares fit). Alternative or addition to
    differential evolution for many parameters (6-10 species).

    n_starts     : number of Latin hypercube start points
    extra_starts : further full start vectors, e.g. [differential evolution result]; they come first
    method       : local method, see local_optimization
    n_jobs       : processes; with n_jobs <= number of starts, the starts run in parallel
    Other keyword arguments go to local_optimization. Returns (param_opt, cost_opt).
    """
    from .least_squares_fit import latin_hypercube_starts
    lhs = latin_hypercube_starts(calibr_setup['param_bnds'], n_starts, seed=seed, log_scale=log_scale)
    starts = [np.asarray(x, dtype=float) for x in (extra_starts or [])] + lhs
    print(f'Multistart: {len(starts)} starts ({len(extra_starts or [])} given, {n_starts} Latin hypercube), '
          f'local method {method}', flush=True)
    return local_optimization(cost_func, starts[0], calibr_setup, maxiter=maxiter, n_restarts=1, seed=seed,
                              history_file=history_file, n_jobs=n_jobs, method=method, log_scale=log_scale,
                              start_points=starts[1:], **kwargs)


def calculate_model_params_direct_local(ll_func, dfs, calibr_setup, rnd_seed=8097):
    """Fit model parameters with a direct local L-BFGS-B optimization."""
    # Use local optimization function
    #np.random.seed(rnd_seed)
    #calibr_setup['param_0'] = [np.random.uniform(*bnd) for bnd in calibr_setup['param_bnds']]
    #calibr_setup['param_0'] += np.random.lognormal(0, .01, size=len(calibr_setup['param_0']))
    data_array = extract_observables_from_df(dfs)
    optim_output = minimize(ll_func, calibr_setup['param_0'], args=(dfs, data_array, calibr_setup['model'],
                                                                    calibr_setup['s_x'], calibr_setup['T_x'],
                                                                    calibr_setup['n_x']),
                            method='L-BFGS-B', tol=1e2, options={'maxiter':15, 'disp': False})#, bounds=calibr_setup['param_bnds'])
    return optim_output.x, optim_output.fun


def calculate_prediction(cost_func, calibr_setup):
    """Optimize prediction parameters and return the best vector and objective value."""
    jac_spasity = mdl.jacobian_sparsity(np.shape(calibr_setup['dfs'][1])[0])
    with open(output_file_predict, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"p{i},"
        f.write(output+"cost\n")
    data_array = extract_observables_from_df(calibr_setup['dfs'])
    calibr_setup['data_array'] = data_array
    optim_output = optimization_func_prediction(cost_func, calibr_setup['param_bnds'], args=(calibr_setup, jac_spasity),
                                     workers=calibr_setup['workers'])
    return optim_output.x, optim_output.fun


# Parameter estimation using minimizstion of the negative log-likelihood function (func)
def optimization_func(func, bnds, args=(), workers=1, maxiter=1000):
    """Run differential evolution with the calibration search settings."""
    return differential_evolution(func, args=args, tol=1e-6, atol=1e-6, maxiter=maxiter, mutation=(0.3, 1.9), recombination=0.7, popsize=30,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  strategy='randtobest1bin', callback=_callback_ll) #init='sobol'

def optimization_func_prediction(func, bnds, args=(), workers=1):
    """Run differential evolution with the prediction search settings."""
    return differential_evolution(func, args=args, tol=1e-3, atol=1e-3, maxiter=300, mutation=(0.3, 1.9), recombination=0.7, popsize=30,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  strategy='randtobest1bin', callback=_callback_ll_predict) #init='sobol'

def _callback_ll(intermediate_result):
    """Saves the best solution and functoin value at each iteration."""
    optimization_history.append((intermediate_result.x.copy(), intermediate_result.fun.copy()))  # Save a copy of x to avoid overwriting
    with open(output_file, "a") as f:
        output = f"{len(optimization_history)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output+f"{intermediate_result.fun}\n")


def _callback_ll_predict(intermediate_result):
    """Saves the best solution and functoin value at each iteration."""
    optimization_history.append((intermediate_result.x.copy(), intermediate_result.fun.copy()))  # Save a copy of x to avoid overwriting
    with open(output_file_predict, "a") as f:                                                       
        output = f"{len(optimization_history)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output+f"{intermediate_result.fun}\n")


############################## Try: 2 step global optimization ###################################
def calculate_model_params_direct_2steps(ll_func, dfs, calibr_setup):
    """Run a two-stage global parameter search, seeding the second stage from the first."""
    jac_spasity = mdl.jacobian_sparsity(np.shape(dfs[1])[0])
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"p{i},"
        f.write(output+"cost\n")

    data_array = extract_observables_from_df(dfs[:-1])
    calibr_setup['data_arr'] = data_array
    solver1 = optimization_func_1step(ll_func, calibr_setup['param_bnds'], args=(calibr_setup, jac_spasity),
                                            workers=calibr_setup['workers'])
    solver1.solve()
    with open(output_file2, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"p{i},"
        f.write(output+"cost\n")
    
    optim_output2 = optimization_func_2step(ll_func, calibr_setup['param_bnds'], init=solver1.population, args=(dfs, data_array, calibr_setup['model'], calibr_setup['s_x'],
                                                                                calibr_setup['T_x'], jac_spasity),
                                            workers=calibr_setup['workers'])
    return optim_output2.x, optim_output2.fun

def optimization_func_1step(func, bnds, args=(), workers=1):
    """Create a first-stage differential-evolution solver for parameter fitting."""
    return DifferentialEvolutionSolver(func, args=args, tol=1e-2, atol=1e-3, maxiter=500, mutation=(1., 1.9), recombination=0.7, popsize=35,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  callback=_callback_ll, strategy='randtobest1bin') #init='sobol'

def optimization_func_2step(func, bnds, init=None, args=(), workers=1):
    """Run the second differential-evolution stage using an optional initial population."""
    return differential_evolution(func, args=args, tol=1e-5, atol=1e-6, maxiter=1000, mutation=(0.3, 1.5), recombination=0.7, popsize=35,
                                  bounds=bnds, init=init, disp=True, polish=False, updating='deferred', workers=workers,
                                  callback=_callback_ll2, strategy='best1bin') #init='sobol'
       
def _callback_ll2(intermediate_result):
    """Saves the best solution and function value at each iteration."""
    optimization_history.append((intermediate_result.x.copy(), intermediate_result.fun.copy()))  # Save a copy of x to avoid overwriting
    #print(f"Iteration {len(optimization_history)}: x = {intermediate_result.x}, f(x) = {intermediate_result.fun}")
    with open(output_file2, "a") as f:
        output = f"{len(optimization_history)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output+f"{intermediate_result.fun}\n") 
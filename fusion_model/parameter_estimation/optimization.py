#!/usr/bin/env python3

import os
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


def calculate_model_params(cost_func, calibr_setup, local_polish=False, local_options=None):
    """
    Global optimization (differential_evolution) of the model parameters.

    local_polish  : if True, the best point found by the global search is refined by a
                    local L-BFGS-B minimization (see local_optimization). The global search
                    stops after few generations and without scipy's own polishing, so its
                    result is usually close to, but not at, a local minimum.
    local_options : dict of keyword arguments for local_optimization
                    (e.g. {'maxiter': 1000, 'n_restarts': 3}).
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
                                     workers=calibr_setup['workers'])
    param_opt, cost_opt = np.array(optim_output.x), optim_output.fun
    if local_polish:
        param_opt, cost_opt = local_optimization(cost_func, param_opt, calibr_setup, jac_spasity=jac_spasity,
                                                 **(local_options or {}))
    return param_opt, cost_opt


# Local refinement of a (global) optimization result
def local_optimization(cost_func, param_init, calibr_setup, jac_spasity=None, maxiter=500, ftol=1e-12, gtol=1e-8,
                       n_restarts=1, jitter_frac=0.02, seed=0, history_file=output_file_local, print_every=10):
    """
    L-BFGS-B minimization of cost_func starting from param_init (e.g. the result of
    differential_evolution) to reach the exact local optimum.

    Only free parameters (lower bound < upper bound) are optimized; fixed ones (e.g. the
    diagonal k_ii = 0 or predefined S entries) stay at their value.
    n_restarts > 1 adds starts from random perturbations of param_init (jitter_frac times the
    bound width); the best result is kept. The cost of every iteration of the best run
    is written to history_file (columns as in optimization_history1.csv).
    Returns (param_opt, cost_opt). The local result is only accepted if it is not worse
    than param_init.
    """
    param_init = np.asarray(param_init, dtype=float)
    bnds = calibr_setup['param_bnds']
    lo = np.array([b[0] for b in bnds], dtype=float)
    hi = np.array([b[1] for b in bnds], dtype=float)
    param_init = np.clip(param_init, lo, hi)
    free = np.where(hi > lo)[0]
    rng = np.random.default_rng(seed)

    def full(z):
        x = param_init.copy()
        x[free] = z
        return x

    def f(z):
        c = cost_func(full(z), calibr_setup, jac_spasity)
        return c if np.isfinite(c) else 1e10

    cost_init = f(param_init[free])
    print(f'Local optimization: start cost = {cost_init:.10g} ({len(free)} free parameters)', flush=True)

    starts = [param_init[free]]
    for _ in range(max(0, n_restarts - 1)):
        z = param_init[free] + rng.normal(0., jitter_frac, size=len(free)) * (hi[free] - lo[free])
        starts.append(np.clip(z, lo[free], hi[free]))

    best = None
    for r, z0 in enumerate(starts):
        history = []

        def callback(intermediate_result):
            history.append((full(intermediate_result.x), intermediate_result.fun))
            if print_every and len(history) % print_every == 0:
                print(f'  [start {r}] iteration {len(history)}: cost = {intermediate_result.fun:.10g}', flush=True)

        res = minimize(f, z0, method='L-BFGS-B', bounds=list(zip(lo[free], hi[free])), callback=callback,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        print(f'  [start {r}] cost = {res.fun:.10g}, iterations = {res.nit}, success = {res.success} ({res.message})',
              flush=True)
        if best is None or res.fun < best[0].fun:
            best = (res, history)

    res, history = best
    if history_file:
        os.makedirs(os.path.dirname(history_file) or '.', exist_ok=True)
        with open(history_file, 'w') as fh:
            fh.write('iteration,' + ''.join(f'p{i},' for i in range(len(param_init))) + 'cost\n')
            for it, (x, c) in enumerate(history, 1):
                fh.write(f'{it},' + ''.join(f'{p},' for p in x) + f'{c}\n')

    if res.fun > cost_init:
        print('Local optimization did not improve the cost; keeping the initial parameters.', flush=True)
        return param_init, cost_init
    print(f'Local optimization: cost {cost_init:.10g} -> {res.fun:.10g}', flush=True)
    return full(res.x), float(res.fun)


def calculate_model_params_direct_local(ll_func, dfs, calibr_setup, rnd_seed=8097):
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
def optimization_func(func, bnds, args=(), workers=1):
    return differential_evolution(func, args=args, tol=1e-6, atol=1e-6, maxiter=10, mutation=(0.3, 1.9), recombination=0.7, popsize=30,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  strategy='randtobest1bin', callback=_callback_ll) #init='sobol'

def optimization_func_prediction(func, bnds, args=(), workers=1):
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
    return DifferentialEvolutionSolver(func, args=args, tol=1e-2, atol=1e-3, maxiter=500, mutation=(1., 1.9), recombination=0.7, popsize=35,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  callback=_callback_ll, strategy='randtobest1bin') #init='sobol'

def optimization_func_2step(func, bnds, init=None, args=(), workers=1):
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
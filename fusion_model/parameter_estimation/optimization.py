#!/usr/bin/env python3

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from .. import model as mdl
from fusion_model.tools.dataframe_functions import extract_observables_from_df


optimization_history = []
output_file = 'out/optimization_history1.csv'
output_file2 = 'out/optimization_history2.csv'
output_file_predict = 'out/optimization_history_predict.csv'


def calculate_model_params(cost_func, calibr_setup):
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
    return np.array(optim_output.x), optim_output.fun


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
    return differential_evolution(func, args=args, tol=1e-6, atol=1e-6, maxiter=3500, mutation=(0.3, 1.9), recombination=0.7, popsize=40,
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
    return differential_evolution(func, args=args, tol=1e-5, atol=1e-6, maxiter=2000, mutation=(0.3, 1.5), recombination=0.7, popsize=35,
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
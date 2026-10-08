"""
Ported from ``fusion_model/parameter_estimation/cost_function.py`` and
``fusion_model/parameter_estimation/optimization.py``.

``cost_arithmetic_mean`` and ``optimization_func`` are needed by
pool_paper_casestudy. ``calculate_model_params`` was previously duplicated
(nearly verbatim) inside ``pool_paper_casestudy/pool_model_functions.py`` --
it is the same generic "log progress to CSV, extract observables, run the
global optimizer" driver as ``fusion_model.parameter_estimation.optimization
.calculate_model_params``, just parametrized here by the observable-
extraction function instead of being hardcoded to fusion_model's
mibi/maldi/ngs triple, so it is common code rather than case-study code.
"""

import numpy as np
from scipy.optimize import differential_evolution

from . import dtf as _dtf

# Path the optimization progress (iteration, parameters, cost) is appended to
# by `_callback_ll` on every generation of the global optimizer. Matches the
# path used by the original `fusion_model.parameter_estimation.optimization`.
output_file = 'out/optimization_history1.csv'

# Keeps a copy of (x, fun) for every callback invocation, exactly like the
# original module-level list in fusion_model.
optimization_history = []


def cost_arithmetic_mean(J_vect):
    """Aggregate a list of residual arrays into a single scalar cost (mean of means)."""
    return np.nanmean([np.nanmean(Ji) for Ji in J_vect])


def _callback_ll(intermediate_result):
    """Saves the best solution and function value at each iteration."""
    optimization_history.append(
        (intermediate_result.x.copy(), intermediate_result.fun.copy())
    )  # Save a copy of x to avoid overwriting
    with open(output_file, "a") as f:
        output = f"{len(optimization_history)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output + f"{intermediate_result.fun}\n")


def optimization_func(func, bnds, args=(), workers=1):
    """Parameter estimation using minimization of the (negative log-likelihood/cost) function."""
    return differential_evolution(
        func, args=args, tol=1e-6, atol=1e-6, maxiter=10, mutation=(0.3, 1.9),
        recombination=0.7, popsize=30, bounds=bnds, init='latinhypercube',
        disp=True, polish=False, updating='deferred', workers=workers,
        strategy='randtobest1bin', callback=_callback_ll,
    )  # init='sobol'


def calculate_model_params(cost_func, calibr_setup, extract_fn=None):
    """Run the global optimizer over `calibr_setup['param_bnds']`, logging progress to `output_file`.

    `extract_fn` defaults to :func:`fusion_core.dtf.extract_observables_from_df`
    (a single wide dataframe `calibr_setup['dfs'] = [df_x]`); pass a different
    callable if a case study's data layout needs a different extraction
    routine.

    Returns `(param_opt, cost_opt)`.
    """
    if extract_fn is None:
        extract_fn = _dtf.extract_observables_from_df
    with open(output_file, "w") as f:
        header = "iteration," + "".join(f"p{i}," for i in range(len(calibr_setup["param_bnds"]))) + "cost\n"
        f.write(header)
    calibr_setup["data_array"] = extract_fn(calibr_setup["dfs"])
    optim_output = optimization_func(
        cost_func,
        calibr_setup["param_bnds"],
        args=(calibr_setup, None),
        workers=calibr_setup["workers"],
    )
    return np.array(optim_output.x), optim_output.fun

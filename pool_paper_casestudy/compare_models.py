import os, sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import fusion_model as fm
from profile_likelihood import count_data_points
from pool_paper_casestudy.pool_model_functions import *
from pool_paper_casestudy.do_local_optim import cost

path2 = "pool_paper_casestudy/out/wo_pH_new/"
n_cl = 4

# ---- load your existing full-Hill-model fit ----
result = fm.output.read_from_json("Result_calibration_5exps_local.json", dir=path2)
param_opt_full = np.array(result["param_ode"])
dfs = pd.read_pickle(path2 + 'dataframe_poolpaper_all.pkl')
exps = sorted(list(set([s.split("_")[0] for s in dfs.columns])))
data_array = extract_observables_from_df([dfs])
x0_vals = param_opt_full[:n_cl*len(exps)]
param_ode_full = param_opt_full[n_cl*len(exps):]   # 16-length: [mu x3, omegaT_lm, k_T_inhib, n, N_texp x3, kappa_T, kappa_LA x6]

calibr_base = {
    "output_path": path2, "n_cl": n_cl, "dfs": [dfs],
    "aggregation_func": fm.pest.cost_arithmetic_mean,
    "exps": exps, "data_array": data_array, "x0": x0_vals,
}

# ---- full Hill model bounds & setup (needed to (re)compute cost_opt_full / n_data) ----
param_ode_bnds_full = tuple(
    [(.2, 1.) for _ in range(3)] +           # mu_opt
    [(0.5, 2.), (1., 8000.), (0.3, 1.5)] +    # omegaT_exp + k_T_inhib + n
    [(8., 9.), (8., 9.), (8., 9.)] +          # N_max_exp
    [(.1, 1.)] +                              # kappa_T
    [(.1, 10)] + [(1., 100.)] +
    [(.1, 10)] + [(1., 100.)] +
    [(.1, 10)] + [(1., 100.)]
)

calibr_full = dict(calibr_base)
calibr_full["model"] = ode_model_coculture_wopH
calibr_full["param_bnds"] = param_ode_bnds_full

cost_opt_full = cost(param_ode_full, calibr_full, None)
n_params_full = len(param_ode_full)
print(f"Full Hill model: cost_opt = {cost_opt_full:.6g}, n_params = {n_params_full}")

# indices of the death-term block within param_ode (16-length vector)
# order: mu(0:3), omegaT_lm(3), k_T_inhib(4), n(5), N_texp(6:9), kappa_T(9), kappa_LA(10:16)
rest_after = param_ode_full[6:]   # everything from N_texp onward, unchanged across all models


def refit(model_func, x0_death, bounds_death, label):
    """Build reduced param_ode (mu's + death-term params + rest), refit, return (cost_opt, n_params, res.x)."""
    mu_part = param_ode_full[:3]
    param_ode_reduced = np.concatenate([mu_part, x0_death, rest_after])

    bounds_reduced = (
        [(.2, 1.) for _ in range(3)] +          # mu_opt (unchanged)
        list(bounds_death) +                     # death-term params (candidate-specific)
        [(8., 9.), (8., 9.), (8., 9.)] +          # N_max_exp
        [(.1, 1.)] +                              # kappa_T
        [(.1, 10)] + [(1., 100.)] +
        [(.1, 10)] + [(1., 100.)] +
        [(.1, 10)] + [(1., 100.)]
    )

    calibr_setup = dict(calibr_base)
    calibr_setup["model"] = model_func
    calibr_setup["param_bnds"] = bounds_reduced

    res = minimize(cost, param_ode_reduced, args=(calibr_setup, None),
                    method="L-BFGS-B", bounds=bounds_reduced,
                    options={"maxiter": 500, "disp": False})

    print(f"{label}: success={res.success}, nit={res.nit}, cost_opt = {res.fun:.6g}, "
          f"n_params = {len(param_ode_reduced)}, "
          f"mu = {res.x[:3]}, death_params = {res.x[3:5]}, N_texp = {res.x[5:8]}")
    return res.fun, len(param_ode_reduced), res.x


# ---- refit each simplified candidate, using curve_fit-derived warm starts ----
cost_mm, n_mm, x_mm = refit(
    ode_model_coculture_wopH_MM,
    x0_death=[0.5108470797227876, 254.79790790361722],   # [omega3, K3]
    bounds_death=[(0.01, 5.), (1., 8000.)],
    label="Michaelis-Menten",
)

cost_exp, n_exp, x_exp = refit(
    ode_model_coculture_wopH_expsat,
    x0_death=[0.45462171090600184, 376.602341918351],    # [omega2, K2]
    bounds_death=[(0.01, 5.), (1., 8000.)],
    label="Exp saturating",
)

# ---- nested likelihood-ratio test vs. full Hill model ----
n_data = count_data_points(param_ode_full, calibr_full, None)
sigma_hat2 = cost_opt_full * n_data / (n_data - n_params_full)
print(f"\nn_data={n_data}, sigma_hat2={sigma_hat2:.6g}")

for label, cost_reduced, n_reduced in [("Michaelis-Menten", cost_mm, n_mm), ("Exp saturating", cost_exp, n_exp)]:
    dof = n_params_full - n_reduced
    LR_stat = (cost_reduced - cost_opt_full) * n_data / sigma_hat2
    p_value = 1 - chi2.cdf(LR_stat, dof)
    verdict = "full Hill model NOT significantly better -- prefer simpler model" if p_value > 0.05 else "full Hill model IS significantly better -- keep it"
    print(f"\n{label}: cost_reduced={cost_reduced:.6g} vs cost_full={cost_opt_full:.6g}")
    print(f"  dof={dof}, LR_stat={LR_stat:.4g}, p={p_value:.4g}  ->  {verdict}")


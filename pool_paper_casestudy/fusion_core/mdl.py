"""
ODE models and model-coupled helpers.

``model_ODE_solution`` was ported from ``fusion_model/model/solving.py``.

Everything else here was moved from
``pool_paper_casestudy/pool_model_functions.py``: the co-culture ODE
right-hand-side functions (``ode_model_coculture*``), the helpers that build
their initial conditions / pH forcing / observables (``set_initial_vals``,
``pH_func``, ``interpolate_series``, ``observable``), and the calibration
``cost`` function together with its per-experiment residual helper
(``sq_diff_oneexp``). ``cost`` is kept next to the models rather than in
``pest.py`` because its per-model "which parameter index to zero out" branch
is written directly against these specific model functions -- it is
model-coupled code, not a generic optimization utility.
"""

import numpy as np
from scipy.integrate import solve_ivp


def model_ODE_solution(model, t, param, x0, const, t0=0., jac=None, jac_spasity=None):
    """Solve the ODE `model` on `t` starting from `x0`, returning the state trajectory."""
    sol_model = solve_ivp(
        model, [t0, t[-1]], x0, dense_output=False, method='LSODA', max_step=0.1,
        t_eval=t, args=(param, x0, const), rtol=1e-5, atol=1e-5, jac=jac,
    )  # , lband=const[1], uband=2*const[1])  # , jac_spasity=jac_spasity
    return sol_model.y


def set_initial_vals(x10, temps, n_cl, pH0=6.):
    """Build a full ODE initial-state vector from the bacterial counts `x10`
    (appends the shared R, T, LA, pH tail state used by every model below)."""
    return np.concatenate((x10, [1., 0., 0., pH0]))


def observable(t, x):
    n = np.array([x[0] + x[1], x[2] + x[3]])
    obs = np.concatenate((n, x[5:]))  # mb add pH
    return obs


def pH_func(t, pH_series):
    # pH_series = [[pH1, t1], [pH2, t2], [pH3, t3], ...] (n_times x 2)
    pH_arr, time_arr = np.array(pH_series).T
    diff = time_arr - t
    return pH_arr[np.argmin(np.abs(diff))]


def interpolate_series(t, pH_series):
    """
    Linearly interpolate a value series at given time(s).

    Parameters
    ----------
    t : float or array-like
        Time point(s) at which to evaluate the series.
    pH_series : array-like, shape (n_times, 2)
        Rows of [value, t], e.g. pH_series = [[pH1, t1], [pH2, t2], ...].
        Does not need to be pre-sorted by time.

    Returns
    -------
    float or np.ndarray
        Interpolated value(s) at t
    """
    series = np.asarray(pH_series, dtype=float)
    values = series[:, 0]
    times = series[:, 1]

    # sort by time in case entries aren't ordered
    order = np.argsort(times)
    times_sorted = times[order]
    values_sorted = values[order]
    return np.interp(t, times_sorted, values_sorted)


def ode_model_coculture(t, x, param, x0, ode_args):
    (x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, _) = x

    (mu_ls_opt, mu_lm_opt,
    pH_ls_min, pH_ls_opt, pH_ls_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omega_ls_exp, omega_lm_exp,
    omegaT_lm_exp, k_T_inhib0, n,
    N_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp,
    q_acid) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls = mu_ls_opt * (pH - pH_ls_min) * (pH_ls_max - pH) / ((pH_ls_opt - pH_ls_min) * (pH_ls_max - pH_ls_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_t = 10**N_texp
    omega_ls = 10**(-3) * omega_ls_exp
    omega_lm = 10**(-3) * omega_lm_exp
    omegaT_lm = omegaT_lm_exp
    kappa_T = 10**(-5) * kappa_T_0
    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm = 10**np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp])

    k_T_inhib = k_T_inhib0
    toxin_death = omegaT_lm * x_lm_sen * T**n / (k_T_inhib**n + T**n)  # omegaT_lm * x_lm_sen * T #

    return [
        (mu_ls * R - omega_ls) * x_ls23K,
        (mu_ls * R - omega_ls) * x_lsCTC494,
        (mu_lm * R  - omega_lm) * x_lm_sen - toxin_death,
        (mu_lm * R  - omega_lm) * x_lm_res,
        -(mu_ls / N_t)*R*x_ls23K - (mu_ls / N_t)*R*x_lsCTC494 - (mu_lm / N_t)*R*x_lm_sen - (mu_lm / N_t)*R*x_lm_res,
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K*x_ls23K + kappa_LA_ls23K_2*R*x_ls23K) + (kappa_LA_lsCTC494*x_lsCTC494 + kappa_LA_lsCTC494_2*R*x_lsCTC494) +
        + kappa_LA_lm  * (x_lm_sen+x_lm_res),  # *R but wo R the curves look better
        0.  # - q_acid * LA
    ]


def ode_model_coculture2(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_ls23K_max,
    pH_lsCTC494_min, pH_lsCTC494_opt, pH_lsCTC494_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omegaT_lm, k_T_inhib, n,
    r_23K,  r_lsCTC494, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls23K = mu_ls23K_opt * (pH - pH_ls23K_min) * (pH_ls23K_max - pH) / ((pH_ls23K_opt - pH_ls23K_min) * (pH_ls23K_max - pH_ls23K_min))
    mu_lsCTC494 = mu_lsCTC494_opt * (pH - pH_lsCTC494_min) * (pH_lsCTC494_max - pH) / ((pH_lsCTC494_opt - pH_lsCTC494_min) * (pH_lsCTC494_max - pH_lsCTC494_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    #print(n, k_T_inhib , T, x_lm_sen)
    toxin_death = omegaT_lm * x_lm_sen * np.abs(T)**n / (k_T_inhib**n + np.abs(T)**n)
    return [
        mu_ls23K * R**r_23K * x_ls23K,
        mu_lsCTC494 * R**r_lsCTC494 * x_lsCTC494,# * (N_ls23K_t /N_lsCTC494_t),
        mu_lm * R * x_lm_sen - toxin_death, #* (N_ls23K_t/N_lm_t)
        mu_lm * R * x_lm_res, # * (N_ls23K_t/N_lm_t),
        #-(mu_ls23K / N_ls23K_t)*R*x_ls23K - (mu_lsCTC494 / N_lsCTC494_t)*R*x_lsCTC494 - (mu_lm / N_lm_t)*R*x_lm_sen - (mu_lm / N_lm_t)*R*x_lm_res,
        -(1/N_lm_t)*(r_23K*mu_ls23K*R**r_23K*x_ls23K + r_lsCTC494*mu_lsCTC494* (R**r_lsCTC494)*x_lsCTC494 + mu_lm*R*(x_lm_sen +
        x_lm_res)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def ode_model_coculture3(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_ls23K_max,
    pH_lsCTC494_min, pH_lsCTC494_opt, pH_lsCTC494_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omegaT_lm, k_T_inhib, n,
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    (pH_cond, n_cl,) = ode_args
    pH = pH_func(t, pH_cond)

    mu_ls23K = mu_ls23K_opt * (pH - pH_ls23K_min) * (pH_ls23K_max - pH) / ((pH_ls23K_opt - pH_ls23K_min) * (pH_ls23K_max - pH_ls23K_min))
    mu_lsCTC494 = mu_lsCTC494_opt * (pH - pH_lsCTC494_min) * (pH_lsCTC494_max - pH) / ((pH_lsCTC494_opt - pH_lsCTC494_min) * (pH_lsCTC494_max - pH_lsCTC494_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    toxin_death = omegaT_lm * x_lm_sen * np.abs(T)**n / (k_T_inhib**n + np.abs(T)**n)
    #kappa_LA_lm_2 = 0. # so it does not decompose LA

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t) + mu_lm*(x_lm_sen +
        x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def ode_model_coculture_wopH(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K, mu_lsCTC494, mu_lm,
    omegaT_lm, k_T_inhib, n,
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    toxin_death = omegaT_lm * x_lm_sen * np.abs(T)**n / (k_T_inhib**n + np.abs(T)**n)
    #kappa_LA_lm_2 = 0. # so it does not decompose LA

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t) + mu_lm*(x_lm_sen +
        x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def ode_model_coculture_wopH_expsat(t, x, param, x0, ode_args):
    """Same as ode_model_coculture_wopH, but death term uses a saturating
    exponential instead of the Hill function -- also drops `n`."""
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K, mu_lsCTC494, mu_lm,
    omega2, K2,
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    toxin_death = omega2 * x_lm_sen * (1 - np.exp(-np.abs(T) / K2))

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t) + mu_lm*(x_lm_sen +
        x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def ode_model_coculture_wopH_MM(t, x, param, x0, ode_args):
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K, mu_lsCTC494, mu_lm,
    omega3, K3_0,                        # <- MM: 2 slots, not 3
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp,
    kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = \
        10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp,
                              kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    K3 = K3_0*100
    toxin_death = omega3 * x_lm_sen * np.abs(T) / (K3 + np.abs(T))   # Michaelis-Menten

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t)
            + mu_lm*(x_lm_sen + x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K
            + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494
            + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen + x_lm_res),
        0.
    ]


def ode_model_coculture_withpH_MM(t, x, param, x0, ode_args):
    #(x_ls23K0, x_lsCTC4940, x_lm_sen0, x_lm_res0, R0, T0, LA0, pH0) = x0
    (x_ls23K, x_lsCTC494, x_lm_sen, x_lm_res, R, T, LA, pH) = x

    (mu_ls23K_opt, mu_lsCTC494_opt, mu_lm_opt,
    pH_ls23K_min, pH_ls23K_opt, pH_ls23K_max,
    pH_lsCTC494_min, pH_lsCTC494_opt, pH_lsCTC494_max,
    pH_lm_min, pH_lm_opt, pH_lm_max,
    omega3, K3_0, 
    N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
    kappa_T_0,
    kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp,
    ) = param

    (pH_cond, n_cl,) = ode_args
    #pH = pH_func(t, pH_cond)
    pH = interpolate_series(t, pH_cond)

    mu_ls23K = mu_ls23K_opt * (pH - pH_ls23K_min) * (pH_ls23K_max - pH) / ((pH_ls23K_opt - pH_ls23K_min) * (pH_ls23K_max - pH_ls23K_min))
    mu_lsCTC494 = mu_lsCTC494_opt * (pH - pH_lsCTC494_min) * (pH_lsCTC494_max - pH) / ((pH_lsCTC494_opt - pH_lsCTC494_min) * (pH_lsCTC494_max - pH_lsCTC494_min))
    mu_lm = mu_lm_opt * (pH - pH_lm_min) * (pH_lm_max - pH) / ((pH_lm_opt - pH_lm_min) * (pH_lm_max - pH_lm_min))

    N_ls23K_t = 10**N_ls23K_texp
    N_lsCTC494_t = 10**N_lsCTC494_texp
    N_lm_t = 10**N_lm_texp
    kappa_T = 10**(-5) * kappa_T_0

    kappa_LA_ls23K, kappa_LA_ls23K_2, kappa_LA_lsCTC494, kappa_LA_lsCTC494_2, kappa_LA_lm, kappa_LA_lm_2 = 10**(-9) * np.array([kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp, kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp, kappa_LA_lm_exp, kappa_LA_lm_2_exp])
    K3 = K3_0*100
    toxin_death = omega3 * x_lm_sen * np.abs(T) / (K3 + np.abs(T))   # Michaelis-Menten

    return [
        mu_ls23K * R * x_ls23K,
        mu_lsCTC494 * R * x_lsCTC494,
        mu_lm * R * x_lm_sen - toxin_death,
        mu_lm * R * x_lm_res,
        -R*(mu_ls23K*x_ls23K*(1/N_ls23K_t) + mu_lsCTC494*x_lsCTC494*(1/N_lsCTC494_t) + mu_lm*(x_lm_sen +
        x_lm_res)*(1/N_lm_t)),
        kappa_T * x_lsCTC494 * R,  #  ??
        (kappa_LA_ls23K + kappa_LA_ls23K_2*R)*x_ls23K + (kappa_LA_lsCTC494 + kappa_LA_lsCTC494_2*R)*x_lsCTC494 + (kappa_LA_lm + kappa_LA_lm_2*R)*(x_lm_sen+x_lm_res),
        0.
    ]


def sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0, param_ode, x_max):
    # TODO mb: do we need to fit also data for BAC, LA (pH)
    # Then obs_x -> obs_x+m
    # + pH instead of temp?
    model = calibr_setup["model"]
    days, [obs_x] = calibr_setup["data_array"]
    #temp = calibr_setup["exp_temps"][exp]
    pH_series = np.array([obs_x[i][-1], days]).T
    const = [pH_series, n_cl]

    C0 = set_initial_vals(x0, None, n_cl, pH0=obs_x[i][-1][0])
    #np.concatenate((np.array(x0), np.array([0., 0.]), np.array([1., 0., 0., 6.])))
    C = model_ODE_solution(model, days, param_ode, C0, const, t0=days[0])
    obs_model = observable(days, C)
    #ll_x0 = (obs_x[i][:-1] - C[:-1]) ** 2 / x_max[:-1]
    ll_x0 = [
        (obs_x[i][0] - obs_model[0]) ** 2 / x_max[0], #  G
        (obs_x[i][1] - obs_model[1]) ** 2 / x_max[1],
        (obs_x[i][2] - obs_model[2]) ** 2 / np.max(x_max[2]), # BAC
        (obs_x[i][3] - obs_model[3]) ** 2,# / np.max(x_max[3]), #/ x_max[3], # LA
        #(obs_x[i][4] - obs_model[4]) ** 2 / np.max(x_max[4]),  # pH
    ]
    return np.array(ll_x0)


def cost(param, calibr_setup, jac_spasity):
    n_cl = calibr_setup["n_cl"]
    exps = calibr_setup["exps"]
    n_exps = len(exps)
    param_ode = param[n_cl*n_exps:]
    param_ode_new = np.copy(param_ode)
    x0_vals = param[:n_cl*n_exps]

    (df_x, ) = calibr_setup["dfs"]
    _, [obs_x] = calibr_setup["data_array"]
    n_cl = calibr_setup['n_cl']  # np.shape(df_maldi)[0]
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    # TODO not clear, should just we compare logaritms?
    x_max = obs_x**2
    x_max[x_max == 0.0] = 1.0
    #ll_x = np.zeros(np.shape(obs_x))
    ll_x = np.zeros(np.shape(obs_x[:, :-1]))
    for i, exp in enumerate(exps):
        #if exp != 'LsCTC494' and exp != 'LsCTC494-Lm' and exp != 'V01' and exp != 'V05':
        if exp != 'LsCTC494-Lm' and exp != 'V05':
            # !! if diff model mu(pH) change 3*n_cl to 2*n_cl !!!
            if calibr_setup['model'] == ode_model_coculture:
                param_ode_new[2*4 + 2 + 3+1] = 0.
            elif calibr_setup['model'] == ode_model_coculture2:
                param_ode_new[4*3+3+3] = 0.
            elif calibr_setup['model'] == ode_model_coculture_wopH:
                param_ode_new[9] = 0.
            elif calibr_setup['model'] == ode_model_coculture_wopH_MM:
                param_ode_new[8] = 0.
            elif calibr_setup['model'] == ode_model_coculture_wopH_expsat:
                param_ode_new[8] = 0.
            elif calibr_setup['model'] == ode_model_coculture_withpH_MM:
                param_ode_new[17] = 0.
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode_new, x_max[i])
        else:
            ll_x[i] = sq_diff_oneexp(calibr_setup, exp, i, n_cl, x0_vals[n_cl*i:n_cl*(i+1)], param_ode, x_max[i])
    ll_x = ll_x[ll_x != 0]
    return calibr_setup["aggregation_func"]([ll_x])

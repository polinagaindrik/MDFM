import numpy as np
from scipy.integrate import solve_ivp, odeint


def calc_obs_model(data, param_x0, calibr_setup, t_model):
    """Simulate each experiment and return bacterial counts and predicted observations."""
    (df_mibi, df_maldi, df_ngs,) = data
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(data[1])[0]
    param_opt = param_x0[n_cl*len(exps):]
    s_x = calibr_setup['s_x']
    obs_mibi = np.zeros((len(exps), len(s_x), len(t_model)))
    obs_maldi = np.zeros((len(exps), len(s_x), np.shape(df_maldi)[0], len(t_model)))
    obs_ngs = np.zeros((len(exps), np.shape(df_ngs)[0], len(t_model)))
    temps = []
    x_count = np.zeros((len(exps) , np.shape(df_maldi)[0], len(t_model)))
    for i, exp in enumerate(exps):
        #df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        #temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        temp = calibr_setup['exp_temps'][exp]
        const = [[temp], n_cl, calibr_setup['media']]
        #C0_opt = 1e3*np.concatenate((param_x0_opt[n_cl*i:n_cl*(i+1)],[1e-3 for _ in range(n_cl)]+[1e-3])) 
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(calibr_setup['model'], t_model, param_opt, C0_opt, const)
        n_C = get_bacterial_count(C, n_cl, 2)
        x_count[i] = n_C
        n_C0 = get_bacterial_count(np.array(C0_opt).reshape(len(C0_opt), 1), n_cl, 2)

        obs_mibi[i] = observable_MiBi(t_model, n_C, calibr_setup['s_x'], n_C0, const, t_model)[1]
        obs_maldi[i] = observable_MALDI(t_model, n_C, calibr_setup['s_x'], n_C0, const, t_model)[1]
        obs_ngs[i] = observable_NGS(t_model, n_C, calibr_setup['T_x'], n_C0, const, t_model)[1]
        temps.append(temp)
    return x_count, obs_mibi, obs_maldi, obs_ngs, temps


# ----------------------------------------------------------------------
# Speed-up: compiled right-hand side for fusion_model2
# ----------------------------------------------------------------------
# ~75% of the time of a cost evaluation is spent in thousands of calls of the Python
# right-hand side. If numba is installed, fusion_model2 (constant temperature) is evaluated
# by a compiled copy with identical results (differences ~1e-12), ~6x faster per solve.
# Settings (module attributes, e.g. fm.mdl.solving.ODE_BACKEND = 'odeint'):
#   USE_NUMBA   : False -> always use the Python model function
#   ODE_BACKEND : 'solve_ivp' (default, LSODA as before) or 'odeint' (same LSODA algorithm called
#                 directly from Fortran, ~2.5x faster again; results differ only at the level
#                 of the solver tolerance, e.g. ~5e-5 in -2 log L)
USE_NUMBA = True
ODE_BACKEND = 'solve_ivp'

try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _fusion_model2_rhs_nb(x, temp, n_cl, param):
        """Compute fusion_model2 derivatives in a Numba-compiled function."""
        lambd = 10.0**param[:n_cl] * temp**param[n_cl:2*n_cl]
        alph = param[2*n_cl:3*n_cl] * temp**param[3*n_cl:4*n_cl]
        nmax = 10.0**param[4*n_cl] * temp**param[4*n_cl+1]
        kij = param[4*n_cl+2:4*n_cl+2+n_cl*n_cl].reshape(n_cl, n_cl) * 5e-8
        xa = x[n_cl:2*n_cl]
        b = kij @ xa
        growth = alph * x[-1] * xa / (1.0 + b)
        out = np.empty_like(x)
        out[:n_cl] = -x[:n_cl] * x[-1] * lambd
        out[n_cl:2*n_cl] = x[:n_cl] * x[-1] * lambd + growth
        out[-1] = -(1.0 / nmax) * np.sum(growth)
        return out
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _rhs(model, param, x0, const):
    """Return an ODE right-hand side, using the compiled model when supported."""
    from .ode_structure import fusion_model2
    temp_cond, n_cl = const[0], const[1]
    if USE_NUMBA and _HAS_NUMBA and model is fusion_model2 and np.size(temp_cond) == 1:
        temp, n_cl = float(np.ravel(temp_cond)[0]), int(n_cl)
        param = np.ascontiguousarray(param, dtype=np.float64)
        return lambda t, x: _fusion_model2_rhs_nb(x, temp, n_cl, param), True
    return (lambda t, x: model(t, x, param, x0, const)), False


# The model solution for given ODEs
def model_ODE_solution(model, t, param, x0, const, t0=0., jac=None, jac_spasity=None):
    """Integrate the selected model and return its state values at the requested times."""
    # parameters may come as a list (e.g. read from a JSON result file); the models need arrays
    param = np.asarray(param, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)
    f, compiled = _rhs(model, param, x0, const)
    if ODE_BACKEND == 'odeint' and jac is None:
        prepend = t[0] != t0
        t_int = np.concatenate([[t0], t]) if prepend else t
        y = odeint(f, x0, t_int, rtol=1e-5, atol=1e-5, hmax=0.1, tfirst=True)
        return (y[1:] if prepend else y).T
    if compiled and jac is None:
        sol_model = solve_ivp(f, [t0, t[-1]], x0, dense_output=False, method='LSODA', max_step=0.1, t_eval=t,
                              rtol=1e-5, atol=1e-5)
    else:
        sol_model = solve_ivp(model, [t0, t[-1]], x0, dense_output=False, method='LSODA', max_step=0.1, t_eval=t,
                              args=(param, x0, const), rtol=1e-5, atol=1e-5, jac=jac)
        #, lband=const[1], uband=2*const[1])#, jac_spasity=jac_spasity
    return sol_model.y


# Get bacterial count array from bacterial states x
def get_bacterial_count(x, n_cl, n_states):
    """Sum each cluster's state components to obtain its total bacterial count."""
    return np.array([np.sum([x[i+n_cl*j] for j in range (n_states)], axis=0) for i in range (n_cl)])


# Get [fpc(xi), fmrs(xi)]
def media_filtering(t, n, s_x, x0, const):
    """Apply media-specific cluster weights to bacterial counts."""
    (temp_cond, n_cl, media) = const
    f_media = np.array(s_x).reshape(-1, n_cl, 1) 
    x_filt = n.reshape((1,)+np.shape(n))*f_media #(n_media, n_sp, n_times)
    return x_filt


def observable_MiBi(t, n, s_x, x0, const, t_meas, std=0.):
    """Predict MiBi measurements by filtering counts and summing clusters per medium."""
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    f_x = media_filtering(t, n, s_x, x0, const)
    return t_meas, np.sum(f_x, axis=1)


def observable_MiBi_woS(t, n, s_x, x0, const, t_meas, std=0.):
    """Predict unfiltered MiBi measurements by summing cluster counts."""
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    return t_meas, np.sum(n, axis=0)


def observable_MALDI(t, n, s_x, x0, const, t_meas, std=0.):
    """Predict MALDI proportions after applying media weights, noise, and thresholding."""
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    f_x = media_filtering(t, n, s_x, x0, const)
    f_x = f_x + np.random.normal(0., std, size=np.shape(f_x))
    # Remove low-abundance entries before normalizing each measurement to proportions.
    f_x[f_x<=0.005] = 0
    return t_meas, np.array([f/np.sum(f, axis=0) for f in f_x])


def observable_NGS(t, n, T_x, x0, const, t_meas, std=0.):
    """Predict normalized NGS proportions using transfer weights and optional noise."""
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    # Apply the NGS transfer weights, then clip negative noisy values and normalize to proportions.
    f_x = media_filtering(t, n, T_x, x0, const)[0]
    f_x = f_x + np.random.normal(0., std, size=np.shape(f_x))
    f_x[f_x<0] = 0
    return t_meas, np.array(f_x/np.sum(f_x, axis=0))


def observable_x(t, n, s_x, x0, const, t_meas, std=0.):
    """Return bacterial counts at the requested measurement times without filtering."""
    n = np.array([xx for tt, xx in zip(t, n.T) if tt in t_meas]).T
    return t_meas, n
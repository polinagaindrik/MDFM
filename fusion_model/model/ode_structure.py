#!/usr/bin/env python3
import numpy as np
import scipy.sparse


def fusion_model2(t, x, param, x0, const):
    """Compute the temperature-dependent fusion-model derivatives for all states."""
    (temp_cond, n_cl, media) = const
    if np.size(temp_cond) == 1:
        (temp, ) = temp_cond
    else:
        (temp, ) = temp_func(t, temp_cond)
    lambd_exp = param[n_cl:2*n_cl]
    lambd = 10**param[:n_cl]  * temp**lambd_exp

    mu_1 = np.array(param[2*n_cl:3*n_cl])
    mu_exp = np.array(param[3*n_cl:4*n_cl])
    mu = mu_1 * temp**mu_exp

    nmax = 10**param[4*n_cl] * temp**param[4*n_cl+1]
    kij_matrix = param[4*n_cl+2:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    return np.concatenate((-x[:n_cl]*x[-1]*lambd,
                            x[:n_cl]*x[-1]*lambd + mu*x[-1]*x[n_cl:2*n_cl]/(1+b),
                            [-(1./nmax)*np.sum(mu*x[-1]*x[n_cl:2*n_cl]/(1+b))]))


# ----------------------------------------------------------------------
# Inhibitor pool (pool model, Gaindrik et al.): the inhibition acts through accumulated inhibitor loads J_i instead of
# the current abundances. States [x1 (n), x2 (n), J (n), R]; growth of species i: mu_i R x2_i / (1 + kappa J_i),
# kappa = 5e-8 as in fusion_model2, with J_i = sum_j k_ij I_j for the amounts I_j of inhibitor produced by species j:
#   F1 (production by all growing cells, pool paper Eq. 3.8): dJ_i/dt = sum_j k_ij x2_j            (units of k: 1/day)
#   F2 (production by the growing cells while the resource lasts, pool paper Eq. 3.9, F = kappa R G):
#                                                             dJ_i/dt = sum_j k_ij R x2_j        (units of k: 1/day)
#      production stops when the resource is used up (stationary phase); it is not reduced by the inhibition itself
# For k_ij = a_i b_j (shared load) all J_i = a_i I with one inhibitor pool I, dI/dt = sum_j b_j x2_j (F1).
# The parameter vector has the layout of fusion_model2; J(0) = 0 (initial_state).
# ----------------------------------------------------------------------
def _fusion_model2_pool(t, x, param, const, growth_associated):
    (temp_cond, n_cl, media) = const
    if np.size(temp_cond) == 1:
        (temp, ) = temp_cond
    else:
        (temp, ) = temp_func(t, temp_cond)
    lambd = 10**param[:n_cl] * temp**param[n_cl:2*n_cl]
    mu = np.asarray(param[2*n_cl:3*n_cl]) * temp**np.asarray(param[3*n_cl:4*n_cl])
    nmax = 10**param[4*n_cl] * temp**param[4*n_cl+1]
    kij = np.asarray(param[4*n_cl+2:4*n_cl+2+n_cl*n_cl]).reshape(n_cl, n_cl)
    lag, act, J, res = x[:n_cl], x[n_cl:2*n_cl], x[2*n_cl:3*n_cl], x[-1]
    growth = mu * res * act / (1. + 5e-8 * J)
    dJ = kij @ (res * act if growth_associated else act)
    return np.concatenate((-lag*res*lambd, lag*res*lambd + growth, dJ, [-(1./nmax)*np.sum(growth)]))


def fusion_model2_f1(t, x, param, x0, const):
    """fusion_model2 with an inhibitor pool produced by all growing cells (F1, see the block comment above)."""
    return _fusion_model2_pool(t, x, param, const, growth_associated=False)


def fusion_model2_f2(t, x, param, x0, const):
    """fusion_model2 with an inhibitor pool produced by the growing cells times the resource (F2, see above)."""
    return _fusion_model2_pool(t, x, param, const, growth_associated=True)


POOL_MODELS = {'F1': fusion_model2_f1, 'F2': fusion_model2_f2}
INHIBITOR_MODES = ('algebraic', 'F1', 'F2')


def model_for_inhibitor(inhibitor=None):
    """fusion_model2 ('algebraic' or None: load = current abundances) or the inhibitor-pool model 'F1' / 'F2'."""
    if inhibitor in (None, 'algebraic'):
        return fusion_model2
    if inhibitor not in POOL_MODELS:
        raise ValueError(f'unknown inhibitor {inhibitor!r}, expected one of {INHIBITOR_MODES}')
    return POOL_MODELS[inhibitor]


def inhibitor_of(model):
    """'F1', 'F2' or 'algebraic' for a model function."""
    for k, m in POOL_MODELS.items():
        if model is m:
            return k
    return 'algebraic'


def initial_state(model, x_lag0, n_cl, param_ode=None, temp=None):
    """Full initial state of a model: lag cells x_lag0 (linear scale), growing cells 1, inhibitor loads 0 (pool
    models) and the resource R(0) = 1 - sum_i (x1_i(0) + x2_i(0)) / N_t(T), so that R is at all times the free
    fraction of the capacity, R = 1 - sum_i x_i / N_t (conservation law of the pool model). param_ode: ODE parameters
    in the model layout (after ode_params_for_model), temp: temperature; without them R(0) = 1 (earlier behaviour)."""
    x_lag0 = np.asarray(x_lag0, dtype=float)
    extra = np.zeros(n_cl) if inhibitor_of(model) != 'algebraic' else np.zeros(0)
    r0 = 1.
    if param_ode is not None:
        p = np.asarray(param_ode, dtype=float)
        nmax = 10**p[4*n_cl] * float(np.ravel(temp)[0])**p[4*n_cl+1]
        r0 = 1. - (np.sum(x_lag0) + n_cl) / nmax
    return np.concatenate((x_lag0, np.ones(n_cl), extra, [r0]))


def fusion_model_linear(t, x, param, x0, const):
    """Compute fusion-model derivatives with linear temperature dependence."""
    (temp_cond, n_cl, media) = const
    if np.size(temp_cond) == 1:
        (temp, ) = temp_cond
    else:
        (temp, ) = temp_func(t, temp_cond)
    #lambd = 1e-3*param[:n_cl] * temp
    lambd = 10**param[:n_cl]  * temp
    #lambd_1 = 10**param[:n_cl]

    mu_1 = np.array(param[n_cl:2*n_cl])
    mu_exp = np.array(param[2*n_cl:3*n_cl])
    mu = mu_1 + mu_exp*temp

    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)

    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    return np.concatenate((-x[:n_cl]*x[-1]*lambd,
                            x[:n_cl]*x[-1]*lambd + mu*x[-1]*x[n_cl:2*n_cl]/(1+b),
                            [-(1./nmax)*np.sum(mu*x[-1]*x[n_cl:2*n_cl]/(1+b))]))


def fusion_model_woR(t, x, param, x0, const):
    """Compute fusion-model derivatives with shared resource limitation in each growth term."""
    (temp_cond, n_cl, media) = const
    #if len(temp_cond) == 1:
    #    (temp, ) = temp_cond
    #else:
    #    temp = temp_func(t, temp_cond)
    (temp, ) = temp_cond
    lambd = 1e-3*param[:n_cl] * temp
    mu = param[n_cl:2*n_cl] + param[2*n_cl:3*n_cl]*temp
    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*1e-7
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    R = 1 - np.sum(x[:2*n_cl])/nmax
    return np.concatenate((-x[:n_cl]*R*lambd, x[:n_cl]*R*lambd + mu*R*x[n_cl:2*n_cl]/(1+b)))


def fusion_model_distr(t, x, param, x0, ode_args): # t, x, param, x0, const
    """Compute derivatives for the distribution-parameterized fusion model."""
    #(temp_cond, n_cl, x0, media, s_x) = ode_args
    (temp_cond, n_cl, media) = ode_args
    lambd = param[:n_cl]
    mu = param[n_cl:2*n_cl]
    nmax = 10**param[2*n_cl]
    k = param[2*n_cl+1:2*n_cl+1+n_cl*(n_cl-1)]
    kij_matrix = np.insert(k, [i*n_cl for i in range(n_cl)], 0).reshape((n_cl, n_cl))*1e-7
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    arr = [
        -x[:n_cl]*x[-1]*lambd,
        x[:n_cl]*x[-1]*lambd + x[n_cl:2*n_cl]*x[-1]*mu/(1+b),
        [-(1./nmax)*np.sum(x[n_cl:2*n_cl]*x[-1]*mu/(1+b))]]
    return np.concatenate(arr)

# For polynomial T-dependence of the parameters
def jacobian_fusion_model(t, x, param, x0, const):
    """Calculate the state Jacobian matrix for the fusion model."""
    (temp_cond, n_cl, media) = const
    (temp, ) = temp_cond
    #lambd = 1e-3*param[:n_cl] * temp
    lambd = 10**param[:n_cl]  * temp
    mu = param[n_cl:2*n_cl] + param[2*n_cl:3*n_cl]*temp
    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)
    #kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(1e-8)
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    dLdx = np.concatenate((-x[-1]*lambd*np.eye(n_cl), np.zeros((n_cl, n_cl)), [-lambd*x[:n_cl]])).T
    dGdx = np.concatenate((x[-1]*lambd*np.eye(n_cl), (x[-1]*mu/(1+b))*np.eye(n_cl), [lambd*x[:n_cl]+mu*x[n_cl:2*n_cl]/(1+b)])).T
    dRdx = np.concatenate((np.zeros((1, n_cl)), [(-x[-1]/nmax)*mu/(1+b)], [[-np.sum(mu*x[n_cl:2*n_cl]/(1+b))/nmax]]), axis=1)
    return np.concatenate((dLdx, dGdx, dRdx), axis=0)


# For polynomial T-dependence of the parameters
def jacobian_sparsity(n_cl):
    """Return the sparse structural pattern of the fusion-model state Jacobian."""
    dLdx = np.concatenate((np.eye(n_cl), np.zeros((n_cl, n_cl)), np.ones((1, n_cl)))).T
    dGdx = np.concatenate((np.eye(n_cl), np.eye(n_cl), np.ones((1, n_cl)))).T
    dRdx = np.concatenate((np.zeros((n_cl, 1)), np.ones((n_cl+1, 1)))).T
    # Convert to sparse matrix
    sparsity_pattern_sparse = scipy.sparse.csr_matrix(np.concatenate((dLdx, dGdx, dRdx), axis=0))
    return sparsity_pattern_sparse


def fusion_model_woT(t, x, param, x0, const):
    """Compute fusion-model derivatives with temperature effects omitted."""
    (temp_cond, n_cl, media) = const
    n_max = 10**param[2*n_cl]*2
    lambd = [5e-3*param[i] for i in range (n_cl)]
    mu0 = [param[i+n_cl]*2 for i in range (n_cl)]
    return [
        -x[i]*x[-1]*(lambd[i]) for i in range (n_cl)
        ] + [
         x[i]*x[-1]*(lambd[i]) +
        (mu0[i])*x[-1]*x[i+n_cl]
        for i in range (n_cl)
        ] + [
        -(1./n_max)*np.sum([(mu0[i])*x[-1]*x[i+n_cl]
        for i in range(n_cl)]),
        ]

# The function that calculates the temperature at each moment of time t with given temperature profile Temp(time)
def temp_func(t, temp_series):
    """Return the supplied temperature associated with the time nearest to t."""
    # temp_series = [[T1, t1], [T2, t2], [T3, t3], ...] (n_times x 2)
    # Use the nearest supplied time point; temperatures are not interpolated.
    temp_arr, time_arr = np.array(temp_series).T
    diff = time_arr - t
    return temp_arr[np.argmin(np.abs(diff))]
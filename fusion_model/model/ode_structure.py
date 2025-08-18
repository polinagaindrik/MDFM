#!/usr/bin/env python3
import numpy as np
import scipy.sparse


def fusion_model2(t, x, param, x0, const):
    (temp_cond, n_cl) = const
    if np.size(temp_cond) == 1:
        (temp, ) = temp_cond
    else:
        (temp, ) = temp_func(t, temp_cond)
    lambd_exp = param[n_cl:2*n_cl]
    lambd = 10**param[:n_cl]  * temp**lambd_exp

    alph_1 = np.array(param[2*n_cl:3*n_cl])
    alph_exp = np.array(param[3*n_cl:4*n_cl])
    alph = alph_1 * temp**alph_exp

    nmax = 10**param[4*n_cl] * temp**param[4*n_cl+1]
    kij_matrix = param[4*n_cl+2:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    return np.concatenate((-x[:n_cl]*x[-1]*lambd,
                            x[:n_cl]*x[-1]*lambd + alph*x[-1]*x[n_cl:2*n_cl]/(1+b),
                            [-(1./nmax)*np.sum(alph*x[-1]*x[n_cl:2*n_cl]/(1+b))]))


def fusion_model_linear(t, x, param, x0, const):
    (temp_cond, n_cl) = const
    if np.size(temp_cond) == 1:
        (temp, ) = temp_cond
    else:
        (temp, ) = temp_func(t, temp_cond)
    #lambd = 1e-3*param[:n_cl] * temp
    lambd = 10**param[:n_cl]  * temp
    #lambd_1 = 10**param[:n_cl]

    alph_1 = np.array(param[n_cl:2*n_cl])
    alph_exp = np.array(param[2*n_cl:3*n_cl])
    alph = alph_1 + alph_exp*temp

    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)

    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    return np.concatenate((-x[:n_cl]*x[-1]*lambd,
                            x[:n_cl]*x[-1]*lambd + alph*x[-1]*x[n_cl:2*n_cl]/(1+b),
                            [-(1./nmax)*np.sum(alph*x[-1]*x[n_cl:2*n_cl]/(1+b))]))


def fusion_model_woR(t, x, param, x0, const):
    (temp_cond, n_cl) = const
    #if len(temp_cond) == 1:
    #    (temp, ) = temp_cond
    #else:
    #    temp = temp_func(t, temp_cond)
    (temp, ) = temp_cond
    lambd = 1e-3*param[:n_cl] * temp
    alph = param[n_cl:2*n_cl] + param[2*n_cl:3*n_cl]*temp
    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*1e-7
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    R = 1 - np.sum(x[:2*n_cl])/nmax
    return np.concatenate((-x[:n_cl]*R*lambd, x[:n_cl]*R*lambd + alph*R*x[n_cl:2*n_cl]/(1+b)))


# For polynomial T-dependence of the parameters
def jacobian_fusion_model(t, x, param, x0, const):
    (temp_cond, n_cl) = const
    (temp, ) = temp_cond
    #lambd = 1e-3*param[:n_cl] * temp
    lambd = 10**param[:n_cl]  * temp
    alph = param[n_cl:2*n_cl] + param[2*n_cl:3*n_cl]*temp
    nmax = 10**param[3*n_cl] * temp
    kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(5e-8) # kij then are sampled between (5e-9, 1e-7)
    #kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(1e-8)
    b = np.sum(kij_matrix*x[n_cl:2*n_cl], axis=1)
    dLdx = np.concatenate((-x[-1]*lambd*np.eye(n_cl), np.zeros((n_cl, n_cl)), [-lambd*x[:n_cl]])).T
    dGdx = np.concatenate((x[-1]*lambd*np.eye(n_cl), (x[-1]*alph/(1+b))*np.eye(n_cl), [lambd*x[:n_cl]+alph*x[n_cl:2*n_cl]/(1+b)])).T
    dRdx = np.concatenate((np.zeros((1, n_cl)), [(-x[-1]/nmax)*alph/(1+b)], [[-np.sum(alph*x[n_cl:2*n_cl]/(1+b))/nmax]]), axis=1)
    return np.concatenate((dLdx, dGdx, dRdx), axis=0)


# For polynomial T-dependence of the parameters
def jacobian_sparsity(n_cl):
    dLdx = np.concatenate((np.eye(n_cl), np.zeros((n_cl, n_cl)), np.ones((1, n_cl)))).T
    dGdx = np.concatenate((np.eye(n_cl), np.eye(n_cl), np.ones((1, n_cl)))).T
    dRdx = np.concatenate((np.zeros((n_cl, 1)), np.ones((n_cl+1, 1)))).T
    # Convert to sparse matrix
    sparsity_pattern_sparse = scipy.sparse.csr_matrix(np.concatenate((dLdx, dGdx, dRdx), axis=0))
    return sparsity_pattern_sparse


def fusion_model_woT(t, x, param, x0, const):
    (temp_cond, n_cl) = const
    n_max = 10**param[2*n_cl]*2
    lambd = [5e-3*param[i] for i in range (n_cl)]
    alph0 = [param[i+n_cl]*2 for i in range (n_cl)]
    return [
        -x[i]*x[-1]*(lambd[i]) for i in range (n_cl)
        ] + [
         x[i]*x[-1]*(lambd[i]) +
        (alph0[i])*x[-1]*x[i+n_cl]
        for i in range (n_cl)
        ] + [
        -(1./n_max)*np.sum([(alph0[i])*x[-1]*x[i+n_cl]
        for i in range(n_cl)]),
        ]

# The function that calculates the temperature at each moment of time t with given temperature profile Temp(time)
def temp_func(t, temp_series):
    # temp_series = [[T1, t1], [T2, t2], [T3, t3], ...] (n_times x 2)
    temp_arr, time_arr = np.array(temp_series).T
    diff = time_arr - t
    return temp_arr[np.argmin(np.abs(diff))]
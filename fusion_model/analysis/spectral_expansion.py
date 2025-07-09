import numpy as np
from scipy.linalg import eigh


# Spectral Expansion methods:
def spectral_expansion(N_order, orth_func, Pvals_func, xbnds=(-1., 1.)):
    x_array = np.linspace(-1, 1, 100)
    #  for given N and oth_func calc eigenvalues and eigen vectors
    res1 = preprocess_se(N_order, orth_func, Bmatrix_legendre, x_array)

    # Calc function values in point defined by eigenvals
    xvals = rescale_axis(res1[0], xbnds)
    Pvals = Pvals_func(xvals) # .reshape()
    if len(np.shape(Pvals)) == 1:
        Pvals = Pvals.reshape((1, len(Pvals)))
    # Calculate the final approximation 
    Pcalc = calc_model_se(*res1, Pvals)
    return rescale_axis(x_array, xbnds), Pcalc, xvals, Pvals


def preprocess_se(N_order, orth_func, Bmatrix_func, x_array):
    B_matrix = Bmatrix_func(N_order)
    eigval, eigvect = eigh(B_matrix)
    eigvect = eigvect.T

    psi, orth_arr = [np.zeros((len(eigval), len(x_array))) for _ in range(2)]
    for n in range (len(eigval)):
        orth_arr[n] = orth_func(n)(x_array)
        orth_arr[n] = orth_arr[n]/np.sqrt(orth_arr[n].dot(orth_arr[n])/len(x_array))

    for k, evec in enumerate(eigvect):
        psi[k] = np.sum([evec[n]*orth_arr[n] for n in range (len(evec))], axis=0)
    return eigval, eigvect, orth_arr, psi


def Bmatrix_legendre(N_order):
    B_matrix = np.zeros((N_order, N_order))
    for n in range (N_order):
        sq_n = np.sqrt(2*n + 1)
        for m in range (N_order):
            sq_m = np.sqrt(2*m + 1)
            B_matrix[n, m] = n / (sq_n*sq_m)*(1-np.min([1., np.abs(n-(m+1))])) + m / (sq_n*sq_m)*(1-np.min([1., np.abs(n-(m-1))]))
    return B_matrix


def calc_model_se(eigval, eigvect, orth_arr, psi, Pvals):
    eigvect0 = eigvect[:, 0]
    term1 = Pvals*eigvect0.reshape((1,)+np.shape(eigvect0))
    return np.sum(term1.reshape(np.shape(term1)+(1,)) * psi.reshape((1,)+np.shape(psi)), axis=-2)


def rescale_axis(x, xbnds):
    (xmin, xmax) = xbnds
    return np.array([(v+1)*0.5*(xmax-xmin)+xmin for v in x])
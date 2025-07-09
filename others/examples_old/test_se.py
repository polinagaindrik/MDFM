import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

from scipy.special import legendre#, hermite
import numpy as np
import matplotlib as plt


def Pmatrix_func_full(eigvals, workers=18, add_name=''):
    temps = eigvals
    n_cl = 3
    n_cl_red = 2
    ntr = 1
    data = fm.data.model_3sp_2media_inhib(temps, ntr, path=path, inhib=True, noise=.0, rel_noise=.0)

    T_x = [1. for _ in range (n_cl)]
    
    s_x =  fm.pest.calc_Smatrix(data, T_x, path=path, workers=18)
    # Save generated data and S matrix
    fm.output.json_dump(s_x.astype(list), 'S_matrix.json', dir=path)
    #s_x = read_from_json(path+'S_matrix.json')

    ########### Find projection matrix for all temperatures together:
    transform_func = fm.dim_red.regular_transform
    Pmatrix, x0_tran, df_ngs_new, df_maldi_new = fm.dim_redcalc_Pmatrix(data, transform_func, n_cl, n_cl_red, s_x, temps=temps,
                                                              workers=workers, path=path, add_name=add_name)
    return np.array(Pmatrix).T


if __name__ == "__main__":
    path='out/test_se/'
    orth_func = legendre
    #P_matrix = lambda x: 1-np.exp(-(x+1)) #1 - x # np.sin(x) #
    '''
    P_matrix = lambda x: np.ones((2, len(x)))*(x.reshape((1, len(x))))**3
    P_matrix = lambda x: np.array([x**3, 4*np.sqrt(np.abs(x))])
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for nterm in range (1, 10):
        temp_array, P_calc, temp_points, P_vals = spectral_expansion(nterm*2, orth_func, P_matrix, xbnds=(-1, 1))
        for i, (pv, pc) in enumerate(zip(P_vals, P_calc)):
            ax[i].plot(temp_array, pc, label='N =' + f' {nterm*2}')
            ax[i].scatter(temp_points, pv)
    P_real = P_matrix(temp_array)
    if len(np.shape(P_real)) == 1:
        P_real = P_real.reshape((1, len(P_real)))
    for i, pr in enumerate(P_real):
        ax[i].plot(temp_array, pr, label='real', color='k', linestyle='dashed')
        ax[i].legend()
        ax[i].set_xlabel('T')
        ax[i].set_ylabel('P')
    plt.savefig(path+f'test_spectr_matrix.png', bbox_inches='tight')
    plt.close(fig)

    P_matrix = lambda x: np.array([[x**3, 4*np.sqrt(np.abs(x))], [x, x]])
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for nterm in range (1, 10):
        temp_array, P_calc, temp_points, P_vals = spectral_expansion(nterm*2, orth_func, P_matrix, xbnds=(-1, 1))
        for i, (pv, pc) in enumerate(zip(P_vals, P_calc)):
            for j, (ppv, ppc) in enumerate(zip(pv, pc)):
                ax[i, j].plot(temp_array, ppc, label='N =' + f' {nterm*2}')
                ax[i, j].scatter(temp_points, ppv)
    P_real = P_matrix(temp_array)
    if len(np.shape(P_real)) == 1:
        P_real = P_real.reshape((1, len(P_real)))
    for i, pr in enumerate(P_real):
        for j, ppr in enumerate(pr):
            ax[i, j].plot(temp_array, ppr, label='real', color='k', linestyle='dashed')
            ax[i, j].legend()
            ax[i, j].set_xlabel('T')
            ax[i, j].set_ylabel('P')
    plt.savefig(path+f'test_spectr_matrix2.png', bbox_inches='tight')
    plt.close(fig)
    exit()
    '''
    '''
    # Try on in-silico data + P matrix estimation:
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for nterm in range (9, 11):
        n_term =  nterm*1
        print('n term = ', n_term)
        temp_array, P_calc, temp_points, P_vals = spectral_expansion(n_term, orth_func, Pmatrix_func_full, xbnds=(2., 14.))
        json_dump({'P_calc': P_calc, 'temp_array': temp_array, 'eigvals': temp_points, 'P_eigvals':P_vals},
                   f'Pmatrix_spectral_expansion_n_{n_term}.json', dir=path)
        for i, (pv, pc) in enumerate(zip(P_vals, P_calc)):
            for j, (ppv, ppc) in enumerate(zip(pv, pc)):
                ax[i, j].plot(temp_array, ppc, label='N =' + f' {n_term}')
                ax[i, j].scatter(temp_points, ppv)
                ax[i, j].legend()
    #P_real = Pmatrix_func_full(temp_array)
    #if len(np.shape(P_real)) == 1:
    #    P_real = P_real.reshape((1, len(P_real)))
    #for j, pr in enumerate(P_real): 
    #    ax[j].plot(temp_array, pr, label='real', color='k', linestyle='dashed')
    #    ax[j].legend()
    #    ax[j].set_xlabel('T')
    #    ax[j].set_ylabel('P')
    plt.savefig(path+f'test_spectr_Pinsilico2.png', bbox_inches='tight')
    plt.close(fig)
    '''


    temp_test = np.array([3., 4., 7., 9., 10., 12., 13.])
    #P_test = Pmatrix_func_full(temp_test, add_name='_test')
    P_test = np.array(fm.output.read_from_json('Pmatrix_temp_separate_test.json', dir=path)['P_matrix']).T
    n_term = int(10)
    res_se = fm.output.read_from_json(f'Pmatrix_spectral_expansion_n_{n_term}.json', dir=path)
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for i, (pt, pc, pv) in enumerate(zip(P_test, res_se['P_calc'], res_se['P_eigvals'])):
        for j, (ppt, ppv, ppc) in enumerate(zip(pt, pv, pc)):
            ax[i, j].plot(res_se['temp_array'], ppc, label=f'model (N={n_term})')
            ax[i, j].scatter(res_se['eigvals'], ppv, label='eigenvalues')
            ax[i, j].scatter(temp_test, ppt, label='test values', color = 'k')
            ax[i, j].legend()
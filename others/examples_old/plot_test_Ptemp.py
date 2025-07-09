import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lin_func(x, k, b):
        return b + k*x

if __name__ == "__main__":
    path = 'out/test_Ptemp/'

    # Read saved data and S matrix
    df_mibi = fm.data.read_pkl('dataframe_mibi.pkl', path)
    df_maldi = fm.data.read_pkl('dataframe_maldi.pkl', path)
    df_ngs = fm.data.read_pkl('dataframe_ngs.pkl', path)
    s_x = fm.output.read_from_json('S_matrix.json', dir=path)
    print('S matrix: \n', s_x)

    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))

    ########### Plot P matrx reults

    ## Extrapolate P(T):

    P_matrices = fm.output.read_from_json('Pmatrix_temp_separate.json', dir=path)['P_matrix']
    Pmatrix= fm.output.read_from_json('Pmatrix_temp_together.json', dir=path)['P_matrix']
    Pinv = np.linalg.pinv(Pmatrix)

    print(np.shape(P_matrices))
    temps = [2., 10., 14.] 
    #temps = [2., 3., 4., 6., 8., 10., 12., 14.]
    #P_matrices[-1] = [P_matrices[-1][1], P_matrices[-1][0]]
    #P_matrices[-2] = [P_matrices[-2][1], P_matrices[-2][0]]
    #P_matrices[-3] = [P_matrices[-3][1], P_matrices[-3][0]]
 
    P_estim = np.zeros((2,)+(np.shape(P_matrices[0])))
    fig, ax = plt.subplots(*np.shape(P_matrices[0]), figsize=(12, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    for i, p_vect in enumerate(zip(*P_matrices)):
        for j, p_vals in enumerate(zip(*p_vect)):
            popt, pcov = curve_fit(lin_func, temps, p_vals)
            P_estim[:, i, j] = popt
            ax[i, j].scatter(temps, p_vals)
            #ax[i, j].plot(np.linspace(1, 15, 50), lin_func(np.linspace(1, 15, 50), *popt))
            ax[i, j].set_xlabel('T')
            ax[i, j].set_ylabel('p')
            ax[i, j].set_ylim(-0.05, 1.05)
            ax[i, j].axhline(y=Pmatrix[i][j], xmin=0, xmax=1, color='k', linestyle='dashed')
    plt.savefig(path+'P_extrapolation.png', bbox_inches='tight')
    plt.close(fig)

    ########### Plot Optimization results for temperatures together:
    
    '''
    res_temp_together = read_from_json('Result_temp_together.json', dir=path)
    n_cl_red = np.shape(Pmatrix)[0]
    model_red = fusion_model2
    calibr_setup_red={
        'n_x': n_cl_red*2+1,
        'model': model_red,
        'P_matrix': res_temp_together['P_matrix'],
        's_x': s_x}
    plot_optimization_result_withP([df_mibi, df_maldi, df_ngs], res_temp_together['param_ode'], calibr_setup_red,
                                   s_x, np.linspace(0, 17, 100), path=path)
    
    ########### Plot Optimization results for temps separate:    
    res_temp_separate = read_from_json('Result_temp_separate.json', dir=path)
    calibr_setup2 = calibr_setup_red
    calibr_setup2['P_matrix'] = res_temp_separate['P_matrix']
    plot_optimization_result_withP_diffT([df_mibi, df_maldi, df_ngs], res_temp_separate['param_ode'], calibr_setup2, s_x,
                                         np.linspace(0, 17, 100), path=path)
    
    '''
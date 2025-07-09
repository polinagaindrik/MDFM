import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np


if __name__ == "__main__":
    path = 'out/test_Ptemp/4spmodel/'
    temps = [2., 10., 14.]
    n_cl = 4
    ntr = 1
    n_media = 2

    df_mibi = fm.output.read_from_json('dataframe_mibi.json', dir=path)
    df_maldi = fm.output.read_from_json('dataframe_maldi.json', dir=path)
    df_ngs = fm.output.read_from_json('dataframe_ngs.json', dir=path)
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    df_maldi, df_ngs, T_x = fm.dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0.001)

    s_x = fm.output.read_from_json(path+'S_matrix.json')
    for exp in exps:
        fm.plotting.plot_filteringS2(exp,  [df_maldi, df_ngs], np.array(s_x), T_x, path=path)
    n_cl_red = 2
    transform_func = fm.dim_red.regular_transform

    ########### Find projection matrix P separately for each temperature:
    res = fm.output.read_from_json('Pmatrix_temp_separate.json', dir=path)
    Pmatrix = np.array(res['P_matrix'])
    x0_tran = np.array(res['x0'])
    
    # Fit model to df:
    model_red = fm.mdl.fusion_model2
    x0_bnds = tuple([(0., 7e3) for _ in range (n_cl_red)] +
                    [(0., 1e2) for _ in range (n_cl_red)] +
                    [(1., 1.)])

    x0_bnds_all = ()
    for _ in range (len(exps)):
        x0_bnds_all+=x0_bnds
    inhib_bnds = [(0., 5.) if i!=j else (0., 0.)
                  for i in range (n_cl_red)
                  for j in range (n_cl_red)]
    
    calibr_setup={
        'model': model_red,
        'param_bnds': x0_bnds_all +
                      tuple([(1e-6, .005) for _ in range (n_cl_red)] +
                            [(.001, 2.)   for _ in range (n_cl_red)] +
                            [(.001, 2.)   for _ in range (n_cl_red)] +
                            [(4., 9.)]                               +
                            inhib_bnds),
        'workers': 17, # number of threads for multiprocessing
        'scaling': 1.,
        'P_matrix': Pmatrix,
        's_x': s_x
    }

    for temp, P, x0 in zip(temps, Pmatrix, x0_tran):
        df_ngs_temp, df_maldi_temp = fm.dtf.filter_dataframe(f'{int(temp):02d}C', [df_ngs, df_maldi])
        Pinv = np.linalg.pinv(P)
        df_ngs_new_temp = transform_func(df_ngs_temp, P, Pinv, x0[0])
        df_maldi_new_temp = fm.dim_red.transform_maldi(transform_func, df_maldi_temp, P, Pinv, x0[1:], s_x)
        fm.plotting.plot_proj_matrix_result(df_ngs_temp,  df_ngs_new_temp, df_maldi_temp, df_maldi_new_temp,
                                path=path, add_name='_Tdepend', media=['PC', 'MRS'])

    # Parameter estimation: diff P for diff T:
    param_opt2 = fm.output.read_from_json('Result_temp_separate.json', dir=path)['param_ode']
    fm.plotting.plot_optimization_result_withP_diffT([df_mibi, df_maldi, df_ngs], param_opt2, calibr_setup, s_x,
                                         np.linspace(0, 17, 100), path=path) 
    fm.plotting.plot_reduced_model([df_mibi, df_maldi, df_ngs], param_opt2, calibr_setup, s_x,
                       np.linspace(0, 17, 100), path=path)
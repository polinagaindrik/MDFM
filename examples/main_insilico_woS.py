import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import time
        

if __name__ == "__main__":
    path = 'out/test_insilicoP_temp_together/'

    # 'Real' model:
    temps = [2., 10., 14.,]
    ntr = 1
    workers = -1

    dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(fm.data.model_13sp_2media_inhib, temps, ntr, inhib=False, noise=0., rel_noise=.0,
                                                               path=path+'13_dim/', cutoff=0., cutoff_prop=0.)
    # Read already geberated dfs from pkl
    #dfs = [pd.read_pickle(path_new+df_name) for df_name in ['dataframe_mibi.pkl', 'dataframe_maldi.pkl', 'dataframe_ngs.pkl']]

    (df_mibi, df_maldi, df_ngs) = dfs
    n_cl = np.shape(dfs[1])[0]
    path_new = path+f'{int(n_cl)}_dim/'
    exps = sorted(list(set([s.split('_')[0] for s in dfs[0].columns])))
    n_media = 2

    start_S = time.time()
    s_x = fm.pest.calc_Smatrix([df_mibi, df_maldi, df_ngs], T_x, s_x_predefined, path=path_new, workers=workers)
    print('S calc time: ', (time.time()-start_S)/60., 'min')
    # Save generated data and S matrix
    fm.output.json_dump(s_x.astype(list), 'S_matrix.json', dir=path_new)
    s_x = fm.output.read_from_json(path_new+'S_matrix.json')
    #plot_filteringS2([df_mibi, df_maldi, df_ngs], np.array(s_x), T_x, exps, path=path_new)

    #s_x = read_from_json(path_new+'S_matrix_true.json')['s_x']

    ########### Find projection matrix for all temperatures together:
    n_cl_red = n_cl
    #transform_func = regular_transform
    #path_new = path+f'{int(n_cl_red)}_dim/'
    
    #Pmatrix, x0_trans, df_ngs_new, df_maldi_new = calc_Pmatrix(dfs, transform_func, n_cl, n_cl_red, s_x,
    #                                                           workers=workers, path=path_new)
    #Pmatrix = read_from_json('Pmatrix_temp_together.json', dir=path_new)['P_matrix']
    #x0 = read_from_json('Pmatrix_temp_together.json', dir=path_new)['x0']
    #df_ngs_new = transform_func(df_ngs, Pmatrix, np.linalg.pinv(Pmatrix), x0[0])
    #df_maldi_new = transform_maldi(transform_func, df_maldi, Pmatrix, np.linalg.pinv(Pmatrix), x0[1:], s_x)
    #plot_proj_matrix_result(df_ngs, df_ngs_new, df_maldi, df_maldi_new, path=path_new, add_name='')
    #exit()

    # Fit model to df:
    model_red = fm.mdl.fusion_model_linear
    x0_bnds = tuple([(1., 4.5) for _ in range (n_cl)])
    x0_bnds_all = ()
    for _ in range (len(exps)):
        x0_bnds_all+=x0_bnds
    inhib_bnds = [(0., 0.) if i!=j else (0., 0.)
                  for i in range (n_cl_red)
                  for j in range (n_cl_red)]
    
    calibr_setup={
        'model': model_red,
        'param_bnds': x0_bnds_all +
                      tuple([(1e-6, .01) for _ in range (n_cl_red)] +
                            [(.1, 1.)    for _ in range (n_cl_red)] +
                            [(0., 1.)    for _ in range (n_cl_red)] +
                            [(6., 9.)]                              +
                            inhib_bnds), # sigma_error
        #'P_matrix': Pmatrix,
        's_x': s_x,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_new,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps,
        'exp_temps': {exp: temp for exp, temp in zip(exps, temps)},
        }

    # Save estimated P and S matrices: 
    #param_opt = calculate_model_prob_withP(log_likelihood_withP, [df_mibi, df_maldi, df_ngs], calibr_setup)


    start = time.time()
    param_opt = fm.pest.calculate_model_params(fm.pest.cost_direct, calibr_setup)[0]
    print((time.time()-start)/60., 'min')

    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together.json', dir=path_new)
    param_opt = fm.output.read_from_json(''+'Result_temp_together.json', dir=path_new)['param_ode']
    #jac_spasity = jacobian_sparsity(np.shape(data[1])[0])
    #print(log_likelihood_direct(np.array(param_opt), data, model_red, s_x, T_x, jac_spasity))
    fm.plotting.plot_optimization_result(np.array(param_opt), calibr_setup, np.linspace(0, 17, 100),
                                         path=path_new, add_name='_calibration')

    '''
    calibr_setup['param_0'] = param_opt
    p = mp.Pool(calibr_setup['workers'])
    N_mult = calibr_setup['workers']
    res = p.starmap(calculate_model_params_direct_local, zip(
            iter.repeat(cost_sum_and_geometric_mean),
            iter.repeat(dfs),
            iter.repeat(calibr_setup),
            [i*7 for i in range(N_mult)]
    ), chunksize=1)
    p.close()
    p.join()
    res = sorted(list(res), key=lambda x: -x[-1])
    param_opt = res[0][0]
    json_dump({'param_ode': param_opt, 's_x': s_x.astype(list), 'T_x': T_x}, 'Result_temp_together_final.json', dir=path_new)
    plot_optimization_result(param_opt, calibr_setup, np.linspace(0, 17, 100), path=path_new, add_name='_calibration')
    '''
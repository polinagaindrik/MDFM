import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np



if __name__ == "__main__":
    path='out/test_realdata/four_dim/'
    workers = 18

    # Read the data:
    exps = [f'V{i:02d}' for i in range(1, 10)]
    cutoff0 = 0.01
    df1 = fm.data.get_all_experiments_dataframe(fm.data.read_ngs1, exps[:7], 'experiments/NGS/')
    df2 = fm.data.get_all_experiments_dataframe(fm.data.read_ngs2, exps[7:], 'experiments/NGS/')
    df_ngs = fm.dtf.merge_dfs([df1, df2])
    df_ngs.to_pickle('experiments/NGS/'+'dataframe_ngs.pkl')
    df_ngs = fm.dtf.drop_column(df_ngs, ['M2'])
    df_ngs_red, bact_ngs = fm.dtf.preprocess_dataframe(df_ngs, cutoff=cutoff0)

    df_maldi = fm.data.get_all_experiments_dataframe(fm.data.read_maldi, exps, 'experiments/MALDI/')
    df_maldi = fm.dtf.drop_column(df_maldi, ['M2', 'VRBD'])
    df_maldi_red, bact_maldi = fm.dtf.preprocess_dataframe(df_maldi, cutoff=cutoff0)

    df_mibi = fm.data.get_all_experiments_dataframe(fm.data.read_mibi, exps, 'experiments/microbiology/')
    df_mibi = fm.dtf.drop_column(df_mibi, ['M2', 'VRBD'])


    ## For now leave only bacteria that is present both in NGS and MALDI:
    #TODO Take into accout that not every ngs genus is present in maldi"

    bact_maldi_ngs = fm.dtf.intersection(bact_maldi, bact_ngs)
    bact_all = sorted(set(list(bact_maldi) + list(bact_ngs)))

    ind_ngs = [bact_all.index(i) for i in list(bact_ngs)]
    T_x = [1. if i in ind_ngs else 0. for i in range (len(bact_all))]

    bact_only_ngs = fm.dtf.rest(bact_all, bact_maldi)
    bact_only_maldi = fm.dtf.rest(bact_all, bact_ngs)

    df_maldi_red = df_maldi_red.T
    df_maldi_red[bact_only_ngs] = np.zeros((len(bact_only_ngs)))
    df_maldi = df_maldi_red.T.sort_values(by=['Genus'], ascending=True)

    df_ngs_red = df_ngs_red.T
    df_ngs_red[bact_only_maldi] = np.zeros((len(bact_only_maldi)))
    df_ngs = df_ngs_red.T.sort_values(by=['Genus'], ascending=True)

    # Estimate S matrix:
 
    #n_media = 2
    #s_x = calc_Smatri(x_old[df_mibi, df_maldi, df_ngs], len(bact_maldi_ngs), n_media, path=path, workers=18)
    #json_dump(s_x.astype(list), 'S_matrix.json', dir=path)
    #print('S matrix: ', s_x)


    # Calculate projection matrix:
    s_x = fm.output.read_from_json('out/test_realdata/'+'S_matrix.json', dir='')
    n_cl_red = 4 # prob can reduce to 4
    #for exp in exps:
    #    plot_filteringS_old(exp, [df_maldi, df_ngs], np.array(s_x),  path=path, clrs=define_bacteria_colors(df_ngs, df_maldi))
        #plot_filteringS_old(exp, [df_maldi, df_ngs], s_x, 4, path=path+f'{exp}_')

    exp_temps = fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    transform_func = fm.dim_red.regular_transform
    temps = sorted(list(set(exp_temps.values())))

    '''
    P_matrices, x0_tran, df_ngs_new, df_maldi_new = calc_Pmatrix([df_mibi, df_maldi, df_ngs], transform_func,
                                                                 len(bact_maldi_ngs), n_cl_red, s_x, temps=temps,
                                                                 workers=workers, path=path)
    for temp, P, x0 in zip(temps, P_matrices, x0_tran):
        df_ngs_temp, df_maldi_temp, df_ngs_new_temp, df_maldi_new_temp = filter_dataframe(f'{int(temp):02d}C',
                                                                                          [df_ngs, df_maldi, df_ngs_new, df_maldi_new])
        print(f'P matrix ({int(temp):02d}C): \n', P)
        print(f'x0 ({int(temp):02d}C): \n', x0)
        plot_proj_matrix_result(df_ngs_temp,  df_ngs_new_temp,  df_maldi_temp,  df_maldi_new_temp,
                                path=path, add_name='_Tdepend')
   
    exit()
    '''


    # Read and plot P matrix result for T separately

    res_T = fm.data.read_from_json('Pmatrix_temp_separate.json', dir=path)
    P_matrices = res_T['P_matrix']
    x0_tran = res_T['x0']
    temps = sorted(list(set(exp_temps.values())))
    '''
    for p, temp in zip(P_matrices, temps):
        pinv = np.linalg.pinv(p)
        df_ngs_temp, df_maldi_temp, df_mibi_temp = filter_dataframe(f'{int(temp):02d}C', [df_ngs, df_maldi, df_mibi])
        df_ngs_new_temp = transform_func(df_ngs_temp, np.array(p), pinv)
        df_maldi_new_temp = transform_maldi(transform_func, df_maldi, p, pinv, x0_tran, s_x)
        plot_proj_matrix_result(df_ngs_temp, df_ngs_new_temp, df_maldi_temp, df_maldi_new_temp,
                                path=path, add_name='_Tdepend')
    '''

   
    # Fit model to df with common Pmatrix:
    model_red = fm.mdl.fusion_model2
    x0_bnds = tuple([(0., 1) for _ in range (n_cl_red)] +
                    [(0., 0.) for _ in range (n_cl_red)] +
                    [(1., 1.)])
    x0_bnds_all = ()
    
    for _ in range (len(exps)):
        x0_bnds_all+=x0_bnds
    inhib_bnds = [(0., 1.) if i!=j else (0., 0.)
                  for i in range (n_cl_red)
                  for j in range (n_cl_red)]
    calibr_setup={
        'n_x': n_cl_red*2+1,
        'model': model_red,
        'param_bnds': x0_bnds_all +
                      tuple([(1e-6, 1.) for _ in range (n_cl_red)] +
                            [(.02, 2.)    for _ in range (n_cl_red)] +
                            [(.02, 2.)    for _ in range (n_cl_red)] +
                            [(7., 10.)]                              +
                            inhib_bnds),
        'workers': 17, # number of threads for multiprocessing
        'scaling': 1.,
        'P_matrix': P_matrices,
        's_x': s_x
    }

    ## Parameter estimation for T dependent P:
    calibr_setup = calibr_setup
    param_opt = fm.dim_red.calculate_model_prob_withP(fm.dim_red.log_likelihood_withP_diffT, [df_mibi, df_maldi, df_ngs], calibr_setup)
    fm.output.json_dump({'param_ode': param_opt.astype(list)}, 'Result_temp_separate.json', dir=path)
    param_opt = fm.output.read_from_json('Result_temp_separate.json', dir=path)['param_ode']
    fm.plotting.plot_optimization_result_withP_diffT([df_mibi, df_maldi, df_ngs], param_opt, calibr_setup, s_x,
                                         np.linspace(0, 17, 100), path=path)
    fm.plotting.plot_reduced_model([df_mibi, df_maldi, df_ngs], param_opt, calibr_setup, s_x,
                       np.linspace(0, 17, 100), path=path)
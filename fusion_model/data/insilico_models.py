#!/usr/bin/env python3

import numpy as np
from ..model import fusion_model2, fusion_model_linear#, jacobian_fusion_model  # noqa: F401
from .output import json_dump
from .data_generation import generate_data_dfs
from ..tools import dataframe_functions as dtf


def prepare_insilico_data(insilico_model, temps, ntr, path='', inhib=False, noise=0., rel_noise=.0, cutoff=0., cutoff_prop=0.):
    data = insilico_model(temps, ntr, path=path, inhib=inhib, noise=noise, rel_noise=rel_noise)
    dfs = data[:-1]
    (df_mibi, df_maldi, df_ngs) = dfs

    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    
    # Take intersection of ngs and maldi bacteria
    '''
    bact_all = dtf.intersection(bact_ngs.values, bact_maldi.values)
    df_maldi = df_maldi.T[bact_all].T
    df_ngs = df_ngs.T[bact_all].T
    T_x = np.ones((len(bact_all)))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    s_x_predefined = np.ones((len(media), len(bact_all)))*np.nan
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined[i, ind_med_null] = 0.
    '''
    # Take union of ngs and maldi bacteria
    df_maldi, df_ngs, T_x, s_x_predefined = dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0.001)
    bact_all = sorted(set(list(bact_maldi) + list(bact_ngs)))
    df_mibi.to_pickle(path+'dataframe_mibi_preprocessed.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi_preprocessed.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs_preprocessed.pkl')
    return [df_mibi, df_maldi, df_ngs], bact_all, T_x, s_x_predefined
  

def model_2sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 2
    s_x = [.5,   1.,    # s_PC
           .7,   .1 ]   # s_MRS
    T_x = [1., 1.] # NGS filtering
    x10_param = [4., 1.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    #np.random.seed(6934113)
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [.001, .003, # lambda
                 .2,   .5,   # alpha0
                 .5,   .3,   # alpha1
                 7.,]  + kij_list
    param_model = param_ode + s_x + T_x
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    df_fullx.to_pickle(path+'dataframe_fullx.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_3sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 3
    s_x = [.5,  1., .5,   # s_PC
           .7,  .1, .6]   # s_MRS        
    T_x = [1., 1., 1.] # NGS filtering
    x10_param = [4., 1., 2.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    np.random.seed(6934113)
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [.001, .003, .005, # lambda
                 .2,   .5, .5,  # alpha0
                 .5,   .3, .4,  # alpha1
                 7.,]  + kij_list
    param_model = param_ode + s_x + T_x
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    df_fullx.to_pickle(path+'dataframe_fullx.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_4sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 4
    s_x = [.5,   1.,   .5,  .5,    # s_PC
           .7,   .1,   .6,  .6]    # s_MRS
    T_x = [0., 1., 1., 1.] # NGS filtering
    x10_param = [4., 1., 2., 2.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    np.random.seed(6934113)
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [.001, .003, .005, .002, # lambda
                 .2,   .5,   .5,    .9,   # alpha0
                 .5,   .3,   .4,    0.,   # alpha1
                 7.,] + kij_list
    param_model = param_ode + s_x + T_x
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_6sp_2media_inhib(temps, ntr, inhib=False, noise=0., rel_noise=0., path=''):
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 6
    np.random.seed(6934113)
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [.003, .005,  .001, .003, .005, .003,  # lambda
                   .5, .5,  .3, .4, .3, .4,  # alpha0
                   .3, .4, .2, .5, .6, .3,   # alpha1
                    7.,] + kij_list 
    s_x = [.0, .4,  .6, .8, 1., .1,  # s_PC
           1., .1, 0., .5, 0., .5]  # s_MRS

    T_x = [0., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10_param = [1., 2., 1., 5., 2., 1.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path_new, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_10sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13.])
    n_cl = 10
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [-5., -3., -2., -3., -6., -3., -2., -4., -5., -3.3, # lambda
                   .2, .5, .5, .5, .3, .4, .3, .4, .2, .5,   # alpha0
                   .6, .3, .4, .5, .2, .5, .6, .3, .6, .3,   # alpha1
                    8.,] + kij_list
    s_x = [.6, .0, .4, .2, .6, .8, 1., .1, 0., .4,   # s_PC
            0., 1., .1, .3, 0., .5, 0., .5, 1., .1]   # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10_param = [1.1, 2., 1.5, 4., 2., 1.05, 2., 1.7, 2., 3.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_10sp_2media_exp(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13.])
    n_cl = 10
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [-5., -3., -2., -3., -6., -3., -2., -4., -5., -3.3, # lambda_1
                  0.,  1.,  2.,  3., 1.5, 2.5,  1., 0.5, 0.8,  0.2, # lambda_exp
                .2, .5, .5, .5, .3, .4, .3, .4, .2, .5,   # alpha0
                .6, .3, .4, .5, .2, .5, .6, .3, .6, 1.3,   # alpha1
                8.,] + kij_list
    s_x = [.6, .0, .4, .2, .6, .8, 1., .1, 0., .4,   # s_PC
            0., 1., .1, .3, 0., .5, 0., .5, 1., .1]   # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10_param = [1.1, 2., 1.5, 4., 2., 1.05, 2., 1.7, 2., 3.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_13sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    t = np.array([0., 1., 3., 6., 10., 13.])
    n_cl = 13
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [.001, .003, .005, .003, .001, .003, .005, .003, .001, .003, .001, .003, .0005, # lambda
                   .1, .4, .4, .4, .2, .04, .2, .3, .1, .4, .1,  .4,   .7,   # alpha0
                   .5, .2, .3, .4, .1, .4,  .5, .2, .5, .2, .2,  .4,   .01,   # alpha1
                    8.,] + kij_list
    s_x = [.6, .0, .4, .2, .6, .8, 1., .1, 0., .4, .5, 0., .5,   # s_PC
           0., 1., .1, .3, 0., .5, 0., .5, 1., .1, .7, .1, .6]   # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10_param = [1., 2., 1., 5., 2., 1., 2., 1., 2., 1., 4., 1., 2.]
    x0 = [10**L0 for L0 in x10_param] + [1. for _ in range (n_cl+1)]
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    df_mibi.to_pickle(path+'dataframe_mibi.pkl')
    df_maldi.to_pickle(path+'dataframe_maldi.pkl')
    df_ngs.to_pickle(path+'dataframe_ngs.pkl')
    df_realx.to_pickle(path+'dataframe_x.pkl')
    json_dump({'param_ode': [x00 for _ in range (len(temps)) for x00 in x10_param]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx
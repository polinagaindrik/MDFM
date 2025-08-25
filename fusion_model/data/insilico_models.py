#!/usr/bin/env python3

import numpy as np
from ..model import fusion_model2, fusion_model_linear#, jacobian_fusion_model
from .output import json_dump, read_from_json
from .data_generation import generate_data_dfs
from ..tools import dataframe_functions as dtf


def generate_x0_for_simulation(n_exp_max=20, n_cl_max=20, path='', add_name=''):
   x10 = np.random.uniform(1., 4., size=(n_exp_max, n_cl_max)) # Try with lognormal distribution
   json_dump({'x0': x10.astype(list)}, path+f'Initial_values_x0{add_name}.json')
   return x10


def get_random_initial_vals(temps, n_cl):
    x10 = np.random.uniform(1., 4., size=(len(temps), n_cl))
    return x10, set_initial_vals(x10, temps, n_cl)


def set_initial_vals(x10, temps, n_cl):
    return [[10**L0 for L0 in x10[i]] + [1. for _ in range (n_cl+1)] for i in range (len(temps))]


def save_all_dfs(dfs, names=[''], path=''):
    for df, n in zip(dfs, names):
        df.to_pickle(path+f'dataframe_{n}.pkl')


def prepare_insilico_data(insilico_model, n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=.0, cutoff=0., cutoff_prop=0., add_name=''):
    data = insilico_model(n_cl, temps, ntr, S_matrix_setup, path=path, inhib=inhib, noise=noise, rel_noise=rel_noise, x10=x10, add_name=add_name)
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
    save_all_dfs([df_mibi, df_maldi, df_ngs], names=[f'mibi{add_name}_preprocessed', f'maldi{add_name}_preprocessed', f'ngs{add_name}_preprocessed'], path=path)
    return [df_mibi, df_maldi, df_ngs], bact_all, T_x, s_x_predefined
  
'''
def model_2sp_2media_inhib(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 2
    s_x = [.5,   1.,    # s_PC
           .7,   .1 ]   # s_MRS
    T_x = [1., 1.] # NGS filtering
    x10 = [[4., 1.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
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
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx, df_fullx],
                 names=['mibi', 'maldi', 'ngs', 'x', 'fullx'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_3sp_2media_inhib(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 3
    s_x = [.5,  1., .5,   # s_PC
           .7,  .1, .6]   # s_MRS
    T_x = [1., 1., 1.] # NGS filtering
    x10 = [[4., 1., 2.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
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
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx, df_fullx],
                 names=['mibi', 'maldi', 'ngs', 'x', 'fullx'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_4sp_2media_inhib(temps, ntr, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 4
    s_x = [.5,   1.,   .5,  .5,    # s_PC
           .7,   .1,   .6,  .6]    # s_MRS
    T_x = [0., 1., 1., 1.] # NGS filtering
    x10 = [[4., 1., 2., 2.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
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
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def model_6sp_2media_inhib(temps, ntr, x10=None, inhib=False, noise=0., rel_noise=0., path=''):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13., 17.])
    n_cl = 6
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
    x10 = [[1., 2., 1., 5., 2., 1.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path_new, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx

def model_6sp_2media_exp(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13.])
    n_cl = 6
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list=[0. for _ in range(n_cl*n_cl)]
    param_ode = [-5., -3., -2., -3., -6., -3., # lambda_1
                  0.,  1.,  2.,  3., 1.5, 2.5, # lambda_exp
                  .2,  .5,  .5,  .5,  .3,  .4, # alpha0
                  .6,  .3,  .4,  .5,  .2,  .5, # alpha1
                  8., 0.5] + kij_list
    s_x = [ .6, .0, .4, .2, .6, .8, 1., .1, 0., .4,   # s_PC
            0., 1., .1, .3, 0., .5, 0., .5, 1., .1]  # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10 = [[1.1, 2., 1.5, 4., 2., 1.05, 2., 1.7, 2., 3.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_10sp_2media_inhib(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
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
                  .2,  .5,  .5,  .5,  .3,  .4,  .3,  .4,  .2,   .5, # alpha0
                  .6,  .3,  .4,  .5,  .2,  .5,  .6,  .3,  .6,   .3, # alpha1
                  8.,] + kij_list
    s_x = [.6, .0, .4, .2, .6, .8, 1., .1, 0., .4,   # s_PC
            0., 1., .1, .3, 0., .5, 0., .5, 1., .1]  # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10 = [[1.1, 2., 1.5, 4., 2., 1.05, 2., 1.7, 2., 3.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_10sp_2media_exp(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
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
                  .2,  .5,  .5,  .5,  .3,  .4,  .3,  .4,  .2,   .1, # alpha0
                  .6,  .3,  .4,  .5,  .2,  .5,  .6,  .3,  .6,  1.3, # alpha1
                  8., 0.5] + kij_list
    s_x = [ .6, .0, .4, .2, .6, .8, 1., .1, 0., .4,  # s_PC
            0., 1., .1, .3, 0., .5, 0., .5, 1., .1]  # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10 =  [[1.1, 2., 1.5, 4., 2., 1.05, 2., 1.7, 2., 3.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_13sp_2media_inhib(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    np.random.seed(4698517)
    t = np.array([0., 1., 3., 6., 10., 13.])
    n_cl = 13
    if inhib:
        kij = np.random.uniform(low=0.0, high=1.0, size=(n_cl, n_cl))
        for i in range (n_cl):
            kij[i, i] = 0.
        kij_list = [kk for k in kij for kk in k]
    else:
        kij_list = [0. for _ in range(n_cl*n_cl)]
    param_ode = [.001, .003, .005, .003, .001, .003, .005, .003, .001, .003, .001, .003, .0005, # lambda
                   .1,   .4,   .4,   .4,   .2,  .04,   .2,   .3,   .1,   .4,   .1,   .4,    .7, # alpha0
                   .5,   .2,   .3,   .4,   .1,  .4,    .5,   .2,   .5,   .2,   .2,   .4,   .01, # alpha1
                   8.,] + kij_list
    s_x = [.6, .0, .4, .2, .6, .8, 1., .1, 0., .4, .5, 0., .5,   # s_PC
           0., 1., .1, .3, 0., .5, 0., .5, 1., .1, .7, .1, .6]   # s_MRS
    T_x = [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    param_model = param_ode + s_x + T_x
    x10 = [[1., 2., 1., 5., 2., 1., 2., 1., 2., 1., 4., 1., 2.] for _ in range (len(temps))]
    x0 = set_initial_vals(x10, temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, df_fullx = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=['mibi', 'maldi', 'ngs', 'x'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+param_ode, 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx
'''

############################# Models for fusion mode paper ##################################
#n_cl_max = 12
#S_general = np.round(np.random.uniform(0.0, 1.0, size=(3, n_cl_max)), 1)
#S_selective = np.round(np.random.uniform(0.0, 05.0, size=(n_cl_max, n_cl_max)) + np.random.uniform(0.6, 0.95, n_cl_max)*np.eye(n_cl_max), 1)
t = np.array([0., 1., 3., 6., 10., 13.])
#np.random.seed(4698517)

def get_res_from_zl2030dict(n_cl, model='linear', dir=''):
    n_cl_zl2030 = 12
    n_exps_zl2030 = 15
    res_zl2030 = read_from_json('Result_calibration_zl2030.json', dir=dir)
    s_x = np.round((np.array(res_zl2030['s_x'])[:, :n_cl]).flatten(), 1)
    T_x = np.array(res_zl2030['T_x'])[:n_cl]
    param_ode_zl2030 = np.round(np.array(res_zl2030['param_ode'])[n_cl_zl2030*n_exps_zl2030:-2*n_cl_zl2030], 1)
    lambd_1 = param_ode_zl2030[:n_cl]
    if model == 'linear':
        alpha0 = param_ode_zl2030[n_cl_zl2030:n_cl_zl2030+n_cl]
        alpha1 = param_ode_zl2030[2*n_cl_zl2030:2*n_cl_zl2030+n_cl]
        n_max_param = param_ode_zl2030[3*n_cl_zl2030:3*n_cl_zl2030+1]
        k_ij = param_ode_zl2030[3*n_cl_zl2030+1:].reshape(n_cl_zl2030, n_cl_zl2030)[:n_cl, :n_cl].flatten()
        param_ode =  np.concatenate([lambd_1, alpha0, alpha1, n_max_param, k_ij])
    elif model == 'exponential':
        lambda_exp = param_ode_zl2030[n_cl_zl2030:n_cl_zl2030+n_cl]
        alpha0 = param_ode_zl2030[2*n_cl_zl2030:2*n_cl_zl2030+n_cl]
        alpha1 = param_ode_zl2030[3*n_cl_zl2030:3*n_cl_zl2030+n_cl]
        n_max_param = param_ode_zl2030[4*n_cl_zl2030:4*n_cl_zl2030+2]
        k_ij = param_ode_zl2030[4*n_cl_zl2030+2:].reshape(n_cl_zl2030, n_cl_zl2030)[:n_cl, :n_cl].flatten()
        param_ode =  np.concatenate([lambd_1, lambda_exp, alpha0, alpha1, n_max_param, k_ij])
    return s_x, T_x, param_ode


# 10 species models (linear model)
def model_2media_linearfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='linear', dir='out/zl2030/linear_model/calibration/')
    s_x = s_x_zl2030
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


############################ For experiments with different number of species ############################
# 10 species model (exponential model)
def model_2media_expfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    np.random.seed(46987)
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='exponential', dir='out/zl2030/exp_model/calibration/')
    s_x = s_x_zl2030
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


################### Experiments with different media (10 species) #######################
def model_1mediaselect_expfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    np.random.seed(46987)
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='exponential', dir='out/zl2030/exp_model/calibration/')
    s_x = s_x_zl2030[:n_cl]
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx


def model_1mediageneral_expfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    np.random.seed(46987)
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='exponential', dir='out/zl2030/exp_model/calibration/')
    s_x = s_x_zl2030[n_cl:]
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model2, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx
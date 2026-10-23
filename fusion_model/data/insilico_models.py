#!/usr/bin/env python3

import numpy as np
from ..model import fusion_model2, fusion_model_linear, fusion_model_distr#, jacobian_fusion_model
from .output import json_dump, read_from_json
from .data_generation import generate_data_dfs
from ..tools import dataframe_functions as dtf


def generate_x0_for_simulation(n_exp_max=20, n_cl_max=20, path='', add_name=''):
   """Generate and save random log-scale initial values for simulated experiments."""
   x10 = np.random.uniform(1., 4., size=(n_exp_max, n_cl_max)) # Try with lognormal distribution
   json_dump({'x0': x10.astype(list)}, path+f'Initial_values_x0{add_name}.json')
   return x10


def get_random_initial_vals(temps, n_cl):
    """Generate random log-scale initial values and their corresponding model states."""
    x10 = np.random.uniform(1., 4., size=(len(temps), n_cl))
    return x10, set_initial_vals(x10, temps, n_cl)


def set_initial_vals(x10, temps, n_cl):
    """Convert log-scale bacterial initial values into the model's full state vectors."""
    return [[10**L0 for L0 in x10[i]] + [1. for _ in range (n_cl+1)] for i in range (len(temps))]


def save_all_dfs(dfs, names=[''], path=''):
    """Save a sequence of dataframes as named pickle files."""
    for df, n in zip(dfs, names):
        df.to_pickle(path+f'dataframe_{n}.pkl')


def prepare_insilico_data(insilico_model, n_cl, temps, ntr, S_matrix_setup, param_ode=None, x10=None, path='', inhib=False, noise=0., rel_noise=.0, cutoff=0., cutoff_prop=0., add_name='', t_meas=None):
    """Generate synthetic observations and align their MALDI and NGS bacteria.
    t_meas: measurement days (whole days; default: module attribute t = 0, 1, ..., 17); only for the
    model_exp based generators (model_1media_exp_*, model_2media_exp*)."""
    extra = {} if t_meas is None else {'t_meas': np.asarray(t_meas, dtype=float)}
    data = insilico_model(n_cl, temps, ntr, S_matrix_setup, param_ode=param_ode, path=path, inhib=inhib, noise=noise, rel_noise=rel_noise, x10=x10, add_name=add_name, **extra)
    dfs = data[:-1]
    (df_mibi, df_maldi, df_ngs) = dfs
    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    # Take intersection of ngs and maldi bacteria
    '''
    bact_all = dtf.intersection(bact_ngs.values, bact_maldi.values)
    df_maldi = df_maldi.T[bact_all].T
    df_ngs = df_ngs.T[bact_all].T
    T_x = np.ones((len(bact_all)))
    s_x_predefined = np.ones((len(media), len(bact_all)))*np.nan
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined[i, ind_med_null] = 0.
    '''
    # Take union of ngs and maldi bacteria
    df_maldi, df_ngs, T_x, s_x_predefined = dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0.001, media=media)
    bact_all = sorted(set(list(bact_maldi) + list(bact_ngs)))
    save_all_dfs([df_mibi, df_maldi, df_ngs], names=[f'mibi{add_name}_preprocessed', f'maldi{add_name}_preprocessed', f'ngs{add_name}_preprocessed'], path=path)
    return [df_mibi, df_maldi, df_ngs], bact_all, T_x, s_x_predefined
  
'''
def model_2sp_2media_inhib(temps, ntr, x10=None, path='', inhib=False, noise=0., rel_noise=0.):
    """Generate two-cluster, two-medium synthetic data using the linear model."""
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
    """Generate three-cluster, two-medium synthetic data using the linear model."""
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
    """Generate four-cluster, two-medium synthetic data using the linear model."""
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
    """Generate six-cluster, two-medium synthetic data with optional inhibition."""
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
    """Generate six-cluster, two-medium synthetic data with exponential temperature effects."""
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
    """Generate ten-cluster, two-medium synthetic data with optional inhibition."""
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
    """Generate ten-cluster, two-medium synthetic data with exponential temperature effects."""
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
    """Generate thirteen-cluster, two-medium synthetic data with optional inhibition."""
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
#t = np.array([0., 1., 3., 6., 10., 13.,])
t = np.linspace(0., 17., 18)
#np.random.seed(4698517)

# ----------------------------------------------------------------------
# True parameters of the in-silico data (model_paper/setup/)
# ----------------------------------------------------------------------
SETUP_DIR = 'model_paper/setup/'                     # relative to the repository root (working directory)
TRUE_PARAMETERS_FILE = 'True_parameters_paper.json'  # written by model_paper/main_true_parameters.py
INITIAL_VALUES_FILE = 'Initial_values_x0_paper.json'
MEDIA_MATRIX_FILE = 'Media_matrix_S_paper.json'


EXP_TEMPS_FILE = 'exp_temps_model_paper.json'  # experiment -> temperature, saved next to the calibration results


def save_exp_temps(exp_temps, dir):
    """Save the temperature of each experiment into the output directory dir."""
    json_dump({str(k): float(v) for k, v in exp_temps.items()}, EXP_TEMPS_FILE, dir=dir)


def read_exp_temps(dir, df=None):
    """
    Temperature of each experiment of the run in dir (exp_temps_model_paper.json saved there by the
    calibration). If the file does not exist (results from before), the temperatures are taken from the
    column names of the data frame df (e.g. 'V01_M1_02C_00_MRS-mibi' -> 2.0).
    """
    import os
    if os.path.exists(os.path.join(dir, EXP_TEMPS_FILE)):
        return read_from_json(EXP_TEMPS_FILE, dir=dir)
    if df is None:
        raise FileNotFoundError(f'{dir}{EXP_TEMPS_FILE} not found; pass the data frame to read the temperatures '
                                f'from its column names')
    return {c.split('_')[0]: float(c.split('_')[2].rstrip('C')) for c in df.columns}


def _split_true_param_ode(param_ode, n_cl_file):
    """[x0 (n_exps x n_cl) | lambda_1 | lambda_exp | alpha_0 | alpha_exp | N_1, N_exp | k (n_cl x n_cl)] -> parts."""
    p = np.asarray(param_ode, dtype=float)
    n = n_cl_file
    n_ode = 4*n + 2 + n*n
    n_x0 = len(p) - n_ode
    if n_x0 < 0 or n_x0 % n:
        raise ValueError(f'param_ode of length {len(p)} does not fit {n} species')
    x0 = p[:n_x0].reshape(-1, n)
    o = p[n_x0:]
    return x0, o[:n], o[n:2*n], o[2*n:3*n], o[3*n:4*n], o[4*n:4*n+2], o[4*n+2:].reshape(n, n)


def read_true_parameters(n_cl, file=TRUE_PARAMETERS_FILE, dir=SETUP_DIR):
    """
    True parameters of the in-silico model for the first n_cl species, from the setup JSON
    (param_ode = [x0 | ODE] of all species, s_x = media x species, T_x). Returns a dict:
      'param_ode': ODE parameters for n_cl species [lambda_1, lambda_exp, alpha_0, alpha_exp, N_1, N_exp, k_ij]
                   (the layout fusion_model2 / model_exp expects)
      'x10'      : log10 initial values, experiments x n_cl
      's_x'      : media x n_cl, 'T_x': n_cl
    """
    res = read_from_json(file, dir=dir)
    n_file = len(res['T_x'])
    if n_cl > n_file:
        raise ValueError(f'{dir}{file} has {n_file} species, {n_cl} requested')
    x0, lam1, lamexp, alph0, alphexp, nmax, k = _split_true_param_ode(res['param_ode'], n_file)
    param_ode = np.concatenate([lam1[:n_cl], lamexp[:n_cl], alph0[:n_cl], alphexp[:n_cl], nmax,
                                k[:n_cl, :n_cl].ravel()])
    s_x = np.asarray(res['s_x'], dtype=float).reshape(-1, n_file)[:, :n_cl]
    return {'param_ode': param_ode, 'x10': x0[:, :n_cl], 's_x': s_x,
            'T_x': np.asarray(res['T_x'], dtype=float)[:n_cl]}


def true_parameters_from_zl2030(zl2030_dir, n_cl=12, n_exps=9, setup_dir=SETUP_DIR,
                                initial_values_file=INITIAL_VALUES_FILE, media_file=MEDIA_MATRIX_FILE):
    """
    Build the true-parameter JSON content (see read_true_parameters) from the ZL2030 calibration
    (zl2030_dir/Result_calibration_zl2030_copy.json, exponential model, ODE parameters rounded to one
    decimal as in get_res_from_zl2030dict), the initial values (first n_exps experiments) and the media
    matrix (s_x = [first selective medium, first general medium], T_x) of the setup folder.
    """
    _, _, ode = get_res_from_zl2030dict(n_cl, model='exponential', dir=zl2030_dir)
    ode = np.array(ode, dtype=float)
    k = ode[4*n_cl+2:].reshape(n_cl, n_cl)
    np.fill_diagonal(k, 0.)  # no self-inhibition
    ode[4*n_cl+2:] = k.ravel()
    x10 = np.array(read_from_json(initial_values_file, dir=setup_dir)['x0'], dtype=float)[:n_exps, :n_cl]
    S = read_from_json(media_file, dir=setup_dir)
    s_x = np.concatenate([np.asarray(S['s_selective'][0][:n_cl], dtype=float),
                          np.asarray(S['s_general'][0][:n_cl], dtype=float)])
    return {'param_ode': list(np.concatenate([x10.ravel(), ode])), 's_x': list(s_x),
            'T_x': list(np.asarray(S['T_x'][:n_cl], dtype=float))}


def get_res_from_zl2030dict(n_cl, model='linear', dir=''):
    """Extract and reshape calibration parameters from ZL2030 project for a requested number of clusters."""
    n_cl_zl2030 = 12
    n_exps_zl2030 = 15 
    res_zl2030 = read_from_json('Result_calibration_zl2030_copy.json', dir=dir)
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
    """Generate two-medium linear-model data using parameters fitted to ZL2030 data."""
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='linear', dir='out/zl2030/linear_model/calibration/')
    s_x = s_x_zl2030
    if not inhib:
        param_ode[-n_cl*n_cl:] = 0.
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    media = ['MRS', 'PC']
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model_linear, t, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                      noise=noise, rel_noise=rel_noise, media=media)#, jac_func=jacobian_fusion_model)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, 'Result_temp_together_real.json', dir=path)
    #plot_insilico_x(df_realx, fusion_model2, t, param_model, x0, n_cl, path=path, add_name=f'{int(n_cl)}sp_insilicodata_')
    return df_mibi, df_maldi, df_ngs, df_realx\


def model_exp(n_cl, temps, ntr, s_x, T_x, media, param_ode=None, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name='', t_meas=None):
    """Generate synthetic observations using the exponential temperature-dependent model."""
    np.random.seed(46987)
    if param_ode is None:  # true ODE parameters from the setup folder (see read_true_parameters)
        param_ode = read_true_parameters(n_cl)['param_ode']
        print(f'model_exp: true ODE parameters of {n_cl} species from {SETUP_DIR}{TRUE_PARAMETERS_FILE}')
    if not inhib:
        param_ode[-n_cl*n_cl:] = 0.
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = get_random_initial_vals(temps, n_cl)
    if t_meas is not None and np.any(np.asarray(t_meas) != np.round(t_meas)):
        raise ValueError('t_meas must be whole days (the data column names contain the day as an integer)')
    df_mibi, df_maldi, df_ngs, df_realx, _ = generate_data_dfs(fusion_model2, t if t_meas is None else t_meas, param_model, x0, temps, n_cl, n_traj=ntr,
                                                                 noise=noise, rel_noise=rel_noise, media=media)
    save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, f'Result_real{add_name}.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


############################ For experiments with different number of species ############################
# Exponential models:
def model_2media_expfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    """Generate two-medium exponential-model data using ZL2030 calibration parameters."""
    np.random.seed(46987)
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='exponential', dir='out/zl2030/exp_model/calibration/')
    s_x = s_x_zl2030
    media = ['MRS', 'PC']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name='')

def model_2media_expgensel(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for general medium 1 and selective medium 1."""
    s_x = np.array(S_matrix_setup["s_selective"][0][:n_cl]+S_matrix_setup["s_general"][0][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel1', 'gen1']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_2media_expgen2sel2(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for general medium 2 and selective medium 2."""
    s_x = np.array(S_matrix_setup["s_selective"][1][:n_cl]+S_matrix_setup["s_general"][1][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel2', 'gen2']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_2media_expgen3sel3(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for general medium 3 and selective medium 3."""
    s_x = np.array(S_matrix_setup["s_selective"][2][:n_cl]+S_matrix_setup["s_general"][2][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel3', 'gen3']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_2media_expsel2sel3(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for selective media 2 and 3."""
    s_x = np.array(S_matrix_setup["s_selective"][1][:n_cl]+S_matrix_setup["s_selective"][2][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel2', 'sel3']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_2media_expsel1sel3(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for selective media 1 and 3."""
    s_x = np.array(S_matrix_setup["s_selective"][0][:n_cl]+S_matrix_setup["s_selective"][2][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel1', 'sel3']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_2media_expsel1sel2(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data for selective media 1 and 2."""
    s_x = np.array(S_matrix_setup["s_selective"][0][:n_cl]+S_matrix_setup["s_selective"][1][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel1', 'sel2']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)


################### Experiments with different media (6 species) #######################
def model_1media_exp_sel1(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data using selective medium 1."""
    s_x = np.array(S_matrix_setup["s_selective"][0][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel1']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_1media_exp_sel2(n_cl, temps, ntr, S_matrix_setup,  **kwargs):
    """Generate exponential-model data using selective medium 2."""
    s_x = np.array(S_matrix_setup["s_selective"][1][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel2']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_1media_exp_sel3(n_cl, temps, ntr, S_matrix_setup,  **kwargs):
    """Generate exponential-model data using selective medium 3."""
    s_x = np.array(S_matrix_setup["s_selective"][2][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['sel3']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_1media_exp_gen1(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data using general medium 1."""
    s_x = np.array(S_matrix_setup["s_general"][0][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['gen1']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_1media_exp_gen2(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data using general medium 2."""
    s_x = np.array(S_matrix_setup["s_general"][1][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['gen2']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_1media_exp_gen3(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    """Generate exponential-model data using general medium 3."""
    s_x = np.array(S_matrix_setup["s_general"][2][:n_cl])
    T_x = S_matrix_setup["T_x"][:n_cl]
    media = ['gen3']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)
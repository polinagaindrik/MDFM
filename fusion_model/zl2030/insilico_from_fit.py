#!/usr/bin/env python3
"""In-silico data generated from a calibration result of the ZL2030 data."""
import numpy as np

from ..data.data_generation import generate_data_dfs
from ..data.insilico_models import get_random_initial_vals, model_exp, save_all_dfs, set_initial_vals, t
from ..data.output import json_dump, read_from_json
from ..model import fusion_model_linear

ZL2030_GENERAL_MEDIUM_DIR = 'out/zl2030/exp_model/calibration/'  # Result_calibration_zl2030_copy.json


def zl2030_general_medium(n_cl=12, dir=ZL2030_GENERAL_MEDIUM_DIR, file='Result_calibration_zl2030_copy.json'):
   """Recovery coefficients of the general medium (plate count agar, PC) estimated from the ZL2030 data
   (second row of s_x, media order ['MRS', 'PC']), first n_cl species."""
   return np.asarray(read_from_json(file, dir=dir)['s_x'], dtype=float)[1, :n_cl]


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


def model_2media_expfromzl2030(n_cl, temps, ntr, S_matrix_setup, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name=''):
    """Generate two-medium exponential-model data using ZL2030 calibration parameters."""
    np.random.seed(46987)
    s_x_zl2030, T_x, param_ode = get_res_from_zl2030dict(n_cl, model='exponential', dir='out/zl2030/exp_model/calibration/')
    s_x = s_x_zl2030
    media = ['MRS', 'PC']
    return model_exp(n_cl, temps, ntr, s_x, T_x, media, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name='')

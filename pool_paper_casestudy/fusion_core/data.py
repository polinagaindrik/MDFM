"""
``save_all_dfs`` is ported from ``fusion_model/data/insilico_models.py``, and
the json helpers are re-exported from :mod:`fusion_core.output`, mirroring
the way ``fusion_model.data`` re-exports ``fusion_model.data.output`` in the
original package (so ``fm.data.json_dump`` and ``fm.output.json_dump`` are
the same function, as before).

Everything else here (in-silico data generation, reading the pool-paper's
Excel sheets, and re-loading a saved calibration run) was moved from
``pool_paper_casestudy/pool_model_functions.py``.
"""

import numpy as np
import pandas as pd

from .output import json_dump, read_from_json  # noqa: F401  (re-exported, same as fusion_model.data)
from . import mdl
from . import dtf


def save_all_dfs(dfs, names=[''], path=''):
    """Pickle each dataframe in `dfs` to `<path>dataframe_<name>.pkl`."""
    for df, n in zip(dfs, names):
        df.to_pickle(path + f'dataframe_{n}.pkl')


########### In-silico data generation ############
def data_generation_poolpaper(n_cl, param_ode, x10, times, path=''):
    dfs_ode = []
    add_name = ''
    temps = [2.,]
    ntr = 1
    df_ode = model_wotemp(n_cl, temps, ntr, times, param_ode=param_ode, x10=x10, add_name=add_name, path=path, exp_start_offset=0)
    dfs_ode.append(df_ode)
    return df_ode


def model_wotemp(n_cl, temps, ntr, times, param_ode=None, x10=None, path='', add_name='', exp_start_offset=0):
    np.random.seed(46987)
    t = times
    if param_ode is None:
       print('No parameter vector provided.')
       exit()
    if x10 is not None:
        x0 = mdl.set_initial_vals(x10, None, n_cl)
    df_ode = generate_data_dfs(mdl.ode_model_coculture, t, np.array(param_ode), x0, temps, n_cl, n_traj=ntr, exp_start_offset=exp_start_offset)
    save_all_dfs([df_ode], names=[f'poolpaper{add_name}'], path=path)
    print(add_name, param_ode, '\n')
    json_dump({'param_ode': [x00 for i in range (len(temps)) for x00 in x10[i]]+list(param_ode)}, f'Generated_param{add_name}.json', dir=path)
    return df_ode


def generate_data_dfs(model, t, param, x0, temps, n_cl, n_traj=1, exp_start_offset=0):
    df_ode = []
    for j, temp in enumerate(temps):
        exp_start = exp_start_offset + 1 + j
        const = [[temp], n_cl]
        x0_exp = np.asarray(x0[j], dtype=float)
        param_ode = np.asarray(param[:n_cl*(4+n_cl)+2])
        x = mdl.model_ODE_solution(model, t, param_ode, x0_exp, const)#, jac=jac)
        bacteria_name = ['Ls', 'Lm']
        df_ode0 = dtf.merge_dfs([dtf.create_df_poolpaper(t, x, [f'V{j+exp_start:02d}'], bacteria_name, stds=0.) for j in range(n_traj)], sort=False)
        df_ode.append(df_ode0)
    return dtf.merge_dfs(df_ode, sort=False)


############### Read data from excel ######################
def experimental_values(name, skiprows=0, path_data='', LA_sheetname='', path='', exp_start_offset=0):
    filename = 'CCD_results_counts_Part 2.xlsx'
    df_counts = pd.read_excel(path_data + filename, keep_default_na=True, sheet_name='R9_rep', skiprows=skiprows, usecols='A:F', nrows=16)
    # TODO: temporal solution to round all t to the round number: maybe not accurate: what else to do?
    time_count = df_counts['Time'].astype(int)

    Ls = np.array(df_counts['LAB (cfu/mL)'])
    Lm = np.array(df_counts['LM (cfu/mL)'])
    pH = np.array(df_counts['pH'])

    if name == 'Lm':
        Ls = np.array([0. for _ in range(len(Lm))])
    elif name == 'Ls23K' or name == 'LsCTC494':
        Lm = np.array([0. for _ in range(len(Ls))])
    elif name == 'LsCTC494-Lm':
        Lm[-1] = np.nan

    if name == 'Ls23K' or name == 'Lm' or name == 'Ls23K-Lm' or name == 'LsCTC494': # ???
        time_BAC = time_count
        BAC = np.array([0. for t in time_BAC])
    else:
        df_BAC = pd.read_excel(path_data+"/8_BA/BA_09.xlsx", keep_default_na=True, sheet_name='BA_prod', skiprows=19, usecols='M:Q', nrows=16)
        time_BAC = df_BAC['Time, h (1)'].astype(int)
        BAC = np.array(df_BAC['BA (10^3 AU/mL)'])*10**3

    df_LA = pd.read_excel(path_data + "/7_LA/RUN_09.xlsx", keep_default_na=True, sheet_name=LA_sheetname, skiprows=19, usecols='M:Q', nrows=16)
    time_LA = df_LA['Time, h (1)'].astype(int)
    # Restore missing measurements of LA
    time_all = sorted(set(list(time_count) + list(time_BAC) + list(time_LA)))
    df_LA["Time, h (1)"] = df_LA["Time, h (1)"].astype(int)
    df_LA_new = df_LA.T
    j = 0
    for t in time_all:
        if not np.any(np.abs(df_LA['Time, h (1)'] - t) <= 0.3):
            j = j+1
            df_LA_new[len(df_LA['Time, h (1)'])+j] = [t]+ [np.nan for _ in range (len(df_LA.columns)-1)]
    df_LA = df_LA_new.T.sort_values(by=['Time, h (1)'], ascending=True)
    LA = np.array(df_LA['LA (mg/mL)'])
    #LA[LA <= 0.0] = 0.0

    #Resource = [np.nan for _ in range (len(time_all))]
    obs = np.array([Ls, Lm, BAC, LA, pH])
    exp_start = exp_start_offset + 1
    df = dtf.create_df_poolpaper(time_all, obs, [f'V{exp_start:02d}'], ['Ls', 'Lm'])
    save_all_dfs([df], names=['poolpaper_' + name], path=path)
    return df


def get_param_dfs(path, path2):
    """Re-load a previous run's optimization-history CSV (best/last row of
    parameters) together with its saved dataframe of experimental data."""
    optim_file2 = "optimization_history1.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    dfs = pd.read_pickle(path2+'dataframe_poolpaper_all.pkl')
    return param_opt, dfs, df_optim2

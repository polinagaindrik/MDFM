import numpy as np
from scipy.optimize import differential_evolution
from ..tools.dataframe_functions import extract_observables_from_df
from ..data.output import json_dump

optimization_history_S = []
output_file = 'out/optimization_history_Smatrix.csv'


def calc_Smatrix(dfs, T_x, s_x_predefined=None, path='', workers=1, add_name=''):
    (df_mibi, df_maldi, df_ngs) = dfs
    bact_all = df_maldi.T.columns
    data_arr = extract_observables_from_df(dfs)
    S_bnds = []
    for i, s in enumerate(s_x_predefined.flatten()):
        if np.isnan(s):
            S_bnds.append((0.05, 1.)) 
        else:
            S_bnds.append((float(s), float(s)))
    
    # Now estimate S matrix by building cost func
    calibr_setup={
        'param_bnds': tuple(S_bnds+[(0.2, 5.)]),
        'T_x_ngs': T_x,
        'cost_func': S_cost_func2, #S_cost_func
        'workers': workers, # number of threads for multiprocessing
    }
    with open(output_file, "w") as f:
        output = "iteration,"
        for i in range(len(calibr_setup['param_bnds'])):
            output += f"s{i//len(bact_all)}{i%len(bact_all)},"
        f.write(output+"cost\n")

    optim_output = optimization_func_Smatrix(calibr_setup['cost_func'], calibr_setup['param_bnds'],
                                     args=(dfs, data_arr, calibr_setup['T_x_ngs']), workers=calibr_setup['workers'])
    print(optim_output.fun)
    # From y_mibi, y_ngs and S_filt calculate \sum x_j for each time point and then real trajectories x_j:
    s0 = optim_output.x[-1]
    print('s0 = ', s0)
    s_x = optim_output.x[:-1].reshape(-1, np.shape(df_maldi)[0])
    if s0 <=1:
        s_x[1] = s0*s_x[1]
    else:
        s_x[0] = s0*s_x[0]

    print('S_filt = ', s_x)
    json_dump({'s_x': s_x.astype(list), 'T_x': T_x}, f'S_matrix{add_name}.json', dir=path)
    return s_x


# Parameter estimation using minimizstion of the negative log-likelihood function (func)
def optimization_func_Smatrix(func, bnds, args=(), workers=1):
    return differential_evolution(func, args=args, tol=1e-8, atol=1e-8, maxiter=1000, mutation=(0.3, 1.9), recombination=0.7, popsize=40,
                                  bounds=bnds, init='latinhypercube', disp=True, polish=False, updating='deferred', workers=workers,
                                  callback=callback_Smatrix, strategy='randtobest1bin') #init='sobol'


def callback_Smatrix(intermediate_result):
    """Saves the best solution and function value at each iteration."""
    optimization_history_S.append((intermediate_result.x.copy(), intermediate_result.fun.copy()))  # Save a copy of x to avoid overwriting
    with open(output_file, "a") as f:
        output = f"{len(optimization_history_S)},"
        for p in intermediate_result.x:
            output += f"{p},"
        f.write(output+f"{intermediate_result.fun}\n")


def calc_S_yngs_array(obs_ngs, s_x):
    obs_ngs = obs_ngs.reshape((1,)+np.shape(obs_ngs))
    s_x = s_x.reshape((np.shape(s_x)[0], np.shape(s_x)[-1], 1))
    return obs_ngs*s_x


def calc_T_ymaldi_array(obs_maldi, T_x):
    T_x = T_x.reshape((1, np.shape(obs_maldi)[-2], 1))
    return obs_maldi*T_x


# Cost function term for S matrix using relation between NGS and MALDI data
def Scost_ngsterm(days, obs_maldi, obs_ngs, s_x, T_x):
    obs_Tmaldi = calc_T_ymaldi_array(np.array(obs_maldi), np.array(T_x))
    Tmaldi_norm = np.sum(obs_Tmaldi, axis=1) # norm over all bacteria
    obs_Sngs = calc_S_yngs_array(obs_ngs, s_x)
    Sngs_norm = np.sum(obs_Sngs, axis=1)
    rhs = Tmaldi_norm.reshape((np.shape(Tmaldi_norm)[0], 1, np.shape(Tmaldi_norm)[1]))*obs_Sngs
    lhs = Sngs_norm.reshape((np.shape(Sngs_norm)[0], 1, np.shape(Sngs_norm)[1]))*obs_Tmaldi
    return rhs, lhs


# Cost function term for scaling between S for diff media s0
def Scost_s0term(days, obs_mibi, obs_maldi, s_x, s0):
    # Leave only data that present in both maldi and ngs
    s1 = s_x[0].reshape((len(s_x[0]), 1))
    s2 = s_x[1].reshape((len(s_x[1]), 1))
    mibi1 = obs_mibi[0].reshape((1, len(obs_mibi[0])))
    mibi2 = obs_mibi[1].reshape((1, len(obs_mibi[1])))
    rhs = np.log(s1*obs_maldi[1]*mibi2+1.) ## ???
    lhs = np.log(s0*s2*obs_maldi[0]*mibi1+1.)
    # TODO to implement more general case of 3 and more media
    return rhs, lhs


# Cost function for S matrix estimation with missing ngs data and scaling between S for different media
def S_squared_difference(params, dfs, data_array, T_x):
    (df_mibi, df_maldi, df_ngs) = dfs
    days_total, [obs_mibi_meas, obs_maldi_meas, obs_ngs_meas] = data_array
    mibi_max = np.nanmax(obs_mibi_meas, axis=(0, 1))
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    s_x = np.array(params[:-1]).reshape(len(media), -1)
    s0 = params[-1]
    sq_diff1 = np.zeros((len(exps), len(media), np.shape(s_x)[-1], len(days_total)))
    sq_diff2 = np.zeros((len(exps), np.shape(s_x)[-1], len(days_total)))
    for i, exp in enumerate(exps):
        rhs1, lhs1 = Scost_ngsterm(days_total, obs_maldi_meas[i], obs_ngs_meas[i], s_x, T_x)
        rhs2, lhs2 = Scost_s0term(days_total, obs_mibi_meas[i]/mibi_max, obs_maldi_meas[i], s_x, s0)
        sq_diff1[i] = (rhs1-lhs1)**2
        sq_diff2[i] = (rhs2-lhs2)**2
    return sq_diff1/np.nansum(s_x**2), sq_diff2/np.nansum(s_x**2)


def S_cost_func(*args, **kwargs):
    sq_diff1, sq_diff2 = S_squared_difference(*args, **kwargs)
    return np.nansum([np.nansum(sq_diff1), np.nansum(sq_diff2)])/(np.size(sq_diff1)+np.size(sq_diff2))


def S_cost_func2(*args, **kwargs):
    a = 0.1
    sq_diff1, sq_diff2 = S_squared_difference(*args, **kwargs)
    num_meas = np.size(sq_diff1[~np.isnan(sq_diff1)]) + np.size(sq_diff2[~np.isnan(sq_diff2)])
    return np.log((np.nansum(np.exp(a*sq_diff1)) + np.nansum(np.exp(a*sq_diff2)))/num_meas)/a
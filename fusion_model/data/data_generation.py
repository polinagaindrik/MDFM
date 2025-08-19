#!/usr/bin/env python3
import numpy as np
import fusion_model.model as mdl
import fusion_model.tools.dataframe_functions as dtf


def generate_data_dfs(model, t, param, x0, temps, n_cl, n_traj=1, jac_func=None, **kwargs):
    df_mibi, df_maldi, df_ngs, df_real, df_fullx = [], [], [], [], []
    for j, temp in enumerate(temps):
        exp_start = 1+j #1+n_traj*j
        if jac_func is not None:
            jac = jac_func#lambda t, x: jac_func(t, x, param[:n_cl*(3+n_cl)+1], x0, [[temp], n_cl])
        else:
            jac = None
        df_mibi0, df_maldi0, df_ngs0, df_realx0, df_fullx0 = generate_insilico_df(model, t, param, x0[j], [[temp], n_cl], n_traj=n_traj,
                                                                                  exp_start_num=exp_start, jac=jac, **kwargs)
        df_mibi.append(df_mibi0)
        df_maldi0[df_maldi0<0] = 0
        df_ngs0[df_ngs0<0] = 0
        df_maldi.append(df_maldi0.apply(dtf.calc_proportion))
        df_ngs.append(df_ngs0.apply(dtf.calc_proportion))
        df_real.append(df_realx0)
        df_fullx.append(df_fullx0)
    df_mibi, df_maldi, df_ngs, df_real, df_fullx = [dtf.merge_dfs(df) for df in [df_mibi, df_maldi, df_ngs, df_real, df_fullx]]
    df_maldi = df_maldi.reindex(sorted(df_maldi.columns), axis=1)
    return df_mibi, df_maldi, df_ngs, df_real, df_fullx


def generate_insilico_df(model, t, param, x0, const,  n_traj=1, exp_start_num=1, n_states=2, jac=None, **kwargs):
    n_cl = const[1]
    x0 = np.asarray(x0, dtype=float)
    if model  == mdl.fusion_model_linear:
        param_ode = np.asarray(param[:n_cl*(3+n_cl)+1])
    else:
        param_ode = np.asarray(param[:n_cl*(4+n_cl)+2])
    x = mdl.model_ODE_solution(model, t, param_ode, x0, const)#, jac=jac)
    count = mdl.get_bacterial_count(x, const[1], n_states)
    create_df = [dtf.create_df_mibi, dtf.create_df_maldi, dtf.create_df_ngs, dtf.create_df_x]
    obss = [mdl.observable_MiBi, mdl.observable_MALDI, mdl.observable_NGS, mdl.observable_x]

    bacteria_name = [f'Bacteria_{i:02d}' for i in range(const[1])]
    df_ode = dtf.merge_dfs([dtf.create_df_fullx(t, x, [f'V{j+exp_start_num:02d}','M1',f'{int(const[0][0]):02d}C'], bacteria_name, stds=0.)
               for j in range(n_traj)])
    return [
        dtf.merge_dfs([insilico_traj_dataframe(t, count, param, x0, const, df_func, bacteria_name,
                                               obs_func=obs, name_part=[f'V{j+exp_start_num:02d}','M1',f'{int(const[0][0]):02d}C'], **kwargs)
                   for j in range(n_traj)])
        for df_func, obs in zip(create_df, obss)
    ] +  [df_ode]


def insilico_traj_dataframe(t, n, param, x0, const, create_df_func, bact_name, name_part=[''], obs_func=None,
                            noise=.0, rel_noise=.0): 
    if obs_func is None:
        observables = n + np.random.normal(0., rel_noise*n+noise, size=np.shape(n))
    else:
        if obs_func == mdl.observable_NGS:
            t_x =  param[-1*const[1]:]
            filt = t_x
        else:
            s_x = param[-3*const[1]:-1*const[1]]
            filt = s_x
        stds = rel_noise*n+noise
        t, observables = obs_func(t, n, filt, x0, const, t, std=stds)
        if obs_func != mdl.observable_x:
            observables = observables + np.random.normal(0., rel_noise*observables+noise, size=np.shape(observables))
        observables[observables<0.00] = 0
    return create_df_func(t, observables, name_part, bact_name, stds=stds)


# The list of possible combinations of parameter combinations, 1D array of probabilities
def sample_parameter_from_distribution(param_flatten, prob_flatten, size=1):
    if len(param_flatten) != 0:
        index_sampled = np.random.choice(np.linspace(0, len(prob_flatten)-1, len(prob_flatten), dtype=int), size=size, 
                                         p=prob_flatten/np.sum(prob_flatten))[0]
        return param_flatten[index_sampled]
    else:
        return param_flatten


# Generate measurement time points fro each trajectory (curve)
def generate_discrete_timepoints(time_bnds, n_times):
    time = np.random.choice(np.linspace(time_bnds[0], time_bnds[1], int(time_bnds[1]-time_bnds[0]+1)),
                            size=(n_times), replace=False)
    return np.sort(time, axis=0)
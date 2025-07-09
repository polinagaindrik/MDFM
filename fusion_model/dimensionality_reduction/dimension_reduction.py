#!/usr/bin/env python3
import numpy as np
import sklearn.cluster as clus
import fusion_model.tools.dataframe_functions as dtf
import fusion_model.model as mdl
from ..data.output import json_dump
from ..parameter_estimation.optimization import optimization_func


def calc_Pmatrix(data, transform_func, n_cl, n_cl_red, s_x, temps=[], workers=1, path='', add_name=''):
    (df_mibi, df_maldi, df_ngs) = data
    p_bnds = [(0., 1.) if j>=i else (0., 0.) for i in range (n_cl_red) for j in range (n_cl)] #+ [(0., 1.) for _ in range (3*n_cl)]
    #p_bnds = [(0., 1.) for i in range (n_cl_red*n_cl)]
    calibr_setup_proj={
        'param_bnds': tuple(p_bnds),
        'workers': workers, # number of threads for multiprocessing
    }   
    if len(temps) == 0:
        Pmatrix, x0_tran, df_ngs_new, df_maldi_new = find_projection_matrix(df_ngs, df_maldi, transform_func, s_x, n_cl_red, calibr_setup_proj)
        json_dump({'P_matrix': Pmatrix.astype(list), 'x0': x0_tran.astype(list)}, 'Pmatrix_temp_together'+add_name+'.json', dir=path)
    else:
        Pmatrix, x0_tran = [], []
        df_ngs_new, df_maldi_new = [], []
        for temp in temps:
            df_ngs_temp,  df_maldi_temp = dtf.filter_dataframe(f'{int(temp):02d}C', [df_ngs, df_maldi])
            P_temp, x0_tr, df_ngs_new_temp, df_maldi_new_temp = find_projection_matrix(df_ngs_temp,  df_maldi_temp, transform_func, s_x, n_cl_red, calibr_setup_proj)
            Pmatrix.append(P_temp)
            x0_tran.append(x0_tr)
            df_ngs_new.append(df_ngs_new_temp)
            df_maldi_new.append(df_maldi_new_temp)
        df_ngs_new = dtf.merge_dfs(df_ngs_new)
        df_maldi_new = dtf.merge_dfs(df_maldi_new)
        json_dump({'P_matrix': Pmatrix, 'x0': x0_tran}, 'Pmatrix_temp_separate'+add_name+'.json', dir=path)
    return Pmatrix, x0_tran, df_ngs_new, df_maldi_new


def clustering_kmeans(df, num_clusters):
    exps = sorted(list(set([s.split('_')[0] for s in df.columns])))
    obs_all = []
    for exp in exps:
        df0 = df.filter(like=exp)
        obs_exp = np.array([df0[col].values.reshape(np.shape(df0[col].values))*5. for col in df0.columns]).T
        obs_all.append(obs_exp)
    obs_all = np.concatenate(obs_all, axis=1)
    X = obs_all
    kmeans = clus.KMeans(n_clusters=num_clusters, random_state=7, n_init=100, init='random', max_iter=500, 
                         tol=1e-6).fit(X)
    y = []
    for n, cl, c in zip(df.T.columns, kmeans.labels_, obs_all):
        y.append((cl, n.split(';')[-4:], c))
    y = sorted(y, key=lambda x: x[0])
    return kmeans, y


def get_new_clusters(df, kmeans):
    clusters_new = [[] for _ in range(np.max(kmeans.labels_)+1)]
    for lab, cl_old in zip(kmeans.labels_, df.T.columns):
        clusters_new[lab].append(cl_old)
    return clusters_new


def clustering_ngs(df_ngs, df_maldi, num_clusters, plot=None):
    kmeans, y = clustering_kmeans(df_ngs, num_clusters)
    clusters_new = get_new_clusters(df_ngs, kmeans)
    df_ngs_new = dtf.get_cluster_dataframe(df_ngs, clusters_new).apply(dtf.calc_proportion)
    df_maldi_new = dtf.get_cluster_dataframe(df_maldi, clusters_new).apply(dtf.calc_proportion)
    if plot is not None:
        plot(df_ngs, kmeans, y)
    return df_ngs_new, df_maldi_new, clusters_new


# Projection matrix estimation, x is a column
def projection_check(x, f, p, pinv):
    prod = np.array(pinv).dot(np.array(p).dot(x))
    return prod#/np.sum(prod)
    #return prod / np.sum(prod)


def substract_initial(x, f, df_ngs0):
    f_new = f[:10]
    return x.values - df_ngs0.filter(like=f_new).T.values[0]


def add_initial(x, f, df_ngs0):
    f_new = f[:10]
    return x.values + df_ngs0.filter(like=f_new).T.values[0]


def affine_transform(df, Pmatrix, Pinv, *args):
    df_new = dtf.update_df(df, substract_initial, df.filter(like='_01_'))
    df_new = dtf.update_df(df_new, projection_check, Pmatrix, Pinv)
    return dtf.update_df(df_new, add_initial, df.filter(like='_01_'))


def affine_transform2(df, Pmatrix, Pinv, x0, *args):
    df_new = dtf.update_df(df, lambda x, f, x0: x.values+x0, x0)
    df_new = dtf.update_df(df_new, projection_check, Pmatrix, Pinv)
    return dtf.update_df(df_new, lambda x, f, x0: x.values+x0, x0)


def transform_maldi(transform_func, df, Pmatrix, Pinv, x0, s_x):
    media = sorted(list(set([samp.split('_')[-1].split('-')[0] for samp in df.columns])))
    df_maldi_new = []
    for x00, s, med in zip(x0, s_x, media):
        Pmaldi = Pmatrix/np.array(s)
        Pmaldiinv = (Pinv.T*np.array(s)).T
        df_maldi_new.append(transform_func(df.filter(like=med), Pmaldi, Pmaldiinv, x00))
    return dtf.merge_dfs(df_maldi_new)


def regular_transform(df, Pmatrix, Pinv, *args):
    return dtf.update_df(df, projection_check, Pmatrix, Pinv)


def P_cost_func(param, transform_func, df_ngs, df_maldi, s_x, n_cl):
    Pmatrix = np.array(param[:n_cl*np.shape(df_ngs)[0]]).reshape(n_cl, -1)
    Pinv = np.linalg.pinv(Pmatrix)
    x0 = param[n_cl*np.shape(df_ngs)[0]:]
    x0 = np.array(x0).reshape(3, -1)

    df_ngs_new = transform_func(df_ngs, Pmatrix, Pinv, x0[0])
    ngs_ll = diversity_data_ll(df_ngs_new, df_ngs, std=1.)

    df_maldi_new = transform_maldi(transform_func, df_maldi, Pmatrix, Pinv, x0[1:], s_x)
    maldi_ll = diversity_data_ll(df_maldi_new, df_maldi, std=1.)
    norm_term = np.sum((np.sum(Pmatrix**2, axis=1) - 1)**2)
    ortog_term = np.sum((Pmatrix.dot(Pmatrix.T) - np.eye(n_cl))**2)
    return (ngs_ll/np.size(df_ngs) + maldi_ll/np.size(df_maldi)) + (ortog_term + norm_term) # 1000*


def find_projection_matrix(df_ngs, df_maldi, transform_func, s_x, n_cl, calibr_setup):
    param_bnds = calibr_setup['param_bnds']
    optim_output = optimization_func(P_cost_func, param_bnds, args=(transform_func, df_ngs, df_maldi, s_x, n_cl),
                                     workers=calibr_setup['workers'])
    print(optim_output.fun)
    Pmatrix = np.array(optim_output.x[:n_cl*np.shape(df_ngs)[0]]).reshape(n_cl, -1)
    x0 = np.array(optim_output.x[n_cl*np.shape(df_ngs)[0]:]).reshape(3, -1)
    Pinv = np.linalg.pinv(Pmatrix)
    df_ngs_new = transform_func(df_ngs, Pmatrix, Pinv, x0[0])

    df_maldi_new = transform_maldi(transform_func, df_maldi, Pmatrix, Pinv, x0[1:], s_x)
    return Pmatrix, x0, df_ngs_new, df_maldi_new


################### Functions With dimension reduction (P matrix)  #######################################333
## With projection matrix P
def calculate_model_prob_withP(ll_func, dfs, calibr_setup):
    optim_output = optimization_func(ll_func, calibr_setup['param_bnds'],
                                     args=(dfs, calibr_setup['model'], calibr_setup['P_matrix'], calibr_setup['s_x']),
                                     workers=calibr_setup['workers'])
    return optim_output.x

def log_likelihood_withP(param_x0, dfs, model, Pmatrix, s_x):
    (df_mibi, df_maldi, df_ngs, ) = dfs
    Pinv = np.linalg.pinv(Pmatrix)
    n_cl = np.shape(Pmatrix)[0]
    maldi_ll, ngs_ll, mibi_ll = 0, 0, 0
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    param = param_x0[(n_cl)*len(exps):]
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = dtf.filter_dataframe(exp, dfs)
        # Calculate model solution + Observables: MiBi, MALDI, NGS
        days = dtf.get_meas_days(df_mibi0, exp)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], n_cl] 
        #C0 = 1e3*np.concatenate((param_x0[n_cl*i:n_cl*(i+1)],[1e-3 for _ in range(n_cl)]+[1e-3]))
        C0 = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        df_mibi_model, df_maldi_model, df_ngs_model = model_one_experiment_withP(df_mibi0, df_maldi0, df_ngs0, model, days, param, C0,
                                                                                 const, Pinv, s_x, dtf.get_meas_days(df_maldi0, exp),
                                                                                 dtf.get_meas_days(df_ngs0, exp), n_states=2)
        df_mibi_model = df_mibi_model.drop("Replicas").apply(calc_log_mibi)
        df_mibi0 = df_mibi0.drop("Replicas").apply(calc_log_mibi)

        df_mibi_ll = df_mibi0.sub(df_mibi_model)**2
        maldi_ll += diversity_data_ll(df_maldi_model, df_maldi0, std=0.05)
        ngs_ll += diversity_data_ll(df_ngs_model, df_ngs0, std=0.05)
        mibi_ll += 10*df_mibi_ll.sum().sum()
    return 0.8*mibi_ll/np.shape(df_mibi)[-1] + 0.1*maldi_ll/(np.size(df_maldi)) + 0.1*ngs_ll/(np.size(df_ngs))


def log_likelihood_withP_diffT(param_x0, dfs, model, Pmatrices, s_x):
    (df_mibi, df_maldi, df_ngs, ) = dfs
    Pinv = [np.linalg.pinv(p) for p in Pmatrices]
    n_cl = np.shape(Pmatrices[0])[0]
    maldi_ll, ngs_ll, mibi_ll = 0, 0, 0
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    param = param_x0[(n_cl)*len(exps):]
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0  = dtf.filter_dataframe(exp, dfs)
        # Calculate model solution + Observables: MiBi, MALDI, NGS
        days = dtf.get_meas_days(df_mibi0, exp)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        if temp == 2.:
            pinv = Pinv[0]
        elif temp == 10.:
            pinv = Pinv[1]
        elif temp == 14.:
            pinv = Pinv[2]
        const = [[temp], n_cl]
        #C0 = 1e3*np.concatenate((param_x0[n_cl*i:n_cl*(i+1)],[1e-3 for _ in range(n_cl)]+[1e-3]))
        C0 = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        df_mibi_model, df_maldi_model, df_ngs_model = model_one_experiment_withP(df_mibi0, df_maldi0, df_ngs0, model, days, param, C0,
                                                                                 const, pinv, s_x, dtf.get_meas_days(df_maldi0, exp),
                                                                                 dtf.get_meas_days(df_ngs0, exp), n_states=2)
        df_mibi_model = df_mibi_model.drop("Replicas").apply(calc_log_mibi)
        df_mibi0 = df_mibi0.drop("Replicas").apply(calc_log_mibi)
        df_mibi_ll = df_mibi0.sub(df_mibi_model)**2
        maldi_ll += diversity_data_ll(df_maldi_model, df_maldi0, std=0.05)
        ngs_ll += diversity_data_ll(df_ngs_model, df_ngs0, std=0.05)
        mibi_ll += df_mibi_ll.sum().sum()
    return mibi_ll/np.shape(df_mibi)[-1] + maldi_ll/(np.size(df_maldi)) + ngs_ll/(np.size(df_ngs))


def model_one_experiment_withP(df_mibi0, df_maldi0, df_ngs0, model, t, param, C0, const, Pinv, s_x, t_obs2,
                               t_obs3, n_states=2):
    C = mdl.model_ODE_solution(model, t, param, C0, const)
    n_C = mdl.get_bacterial_count(C, np.shape(Pinv)[1], n_states)
    n_C0 = mdl.get_bacterial_count(np.array(C0).reshape(len(C0), 1), np.shape(Pinv)[1], n_states)

    n_x = Pinv.dot(n_C)
    n_x0 = Pinv.dot(n_C0)

    const_sp = const
    const_sp[1] = np.shape(Pinv)[0]

    days_mibi, obs_mibi = mdl.observable_MiBi(t, n_x, s_x, n_x0, const_sp, t)
    days_maldi, obs_maldi = mdl.observable_MALDI(t, n_x, s_x, n_x0, const_sp, t_obs2)
    days_ngs, obs_ngs = mdl.observable_NGS(t, n_x, s_x, n_x0, const_sp, t_obs3)

    df_mibi_model = mdl.create_df_mibi(days_mibi, obs_mibi, df_mibi0.columns[0].split('_')[:3], df_mibi0.T.columns)
    df_maldi_model = mdl.create_df_maldi(days_maldi, obs_maldi, df_maldi0.columns[0].split('_')[:3], df_maldi0.T.columns)
    df_ngs_model = mdl.create_df_ngs(days_ngs, obs_ngs, df_ngs0.columns[0].split('_')[:3], df_ngs0.T.columns)
    return df_mibi_model, df_maldi_model, df_ngs_model

def calc_log_mibi(x):
    mean = np.max([x['Average'], 1.])
    return [np.log(mean), np.abs(np.array(x['Standard deviation'])*0.43 / x['Average'])]

def diversity_data_ll(df_model, df_meas, std=0.01):
    df_ll = df_meas.sub(df_model)**2 / std**2
    return df_ll.sum().sum()


def xdata_ll(df_model, df_meas, std=0.1):
    df_ll = np.log(df_meas).sub(np.log(df_model))**2 / std**2
    return df_ll.sum().sum()
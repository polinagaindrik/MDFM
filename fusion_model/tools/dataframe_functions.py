import pandas as pd
import numpy as np


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
    

def rest(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


# Create dataframes for lnown days and observables
def create_df_mibi(days, obs, name_part, bact_name, stds=None):
    if stds is None:
        stds = np.zeros(np.shape(obs))
    data = {'Properties': ['Average', 'Standard deviation', 'Replicas']}
    media = sorted(['MRS', 'PC'])
    for d, o, std in zip(days, obs.T, stds.T):   
        data["_".join(name_part + [f'{int(d):02d}', f'{media[0]}-mibi'])] = [o[0], std[0], [o[0]]]
        data["_".join(name_part + [f'{int(d):02d}', f'{media[1]}-mibi'])] = [o[1], std[1], [o[1]]]
    return pd.DataFrame(data=data).groupby('Properties').sum()


def create_df_sumx(days, obs, name_part, bact_name, stds=None):
    data = {'Properties': ['Average']}
    for d, o in zip(days, obs):   
        data["_".join(name_part + [f'{int(d):02d}', '-sumx'])] = [o]
    return pd.DataFrame(data=data).groupby('Properties').sum()


def create_df_maldi(days, obs, name_part, bact_name, stds=0):
    data = {"Genus": bact_name}
    media = sorted(['MRS', 'PC'])
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', f'{media[0]}-maldi'])] = o.T[0]
        data["_".join(name_part + [f'{int(d):02d}', f'{media[1]}-maldi'])] = o.T[1]
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df.apply(calc_proportion)


def create_df_ngs(days, obs, name_part, bact_name, stds=0):
    data = {"Genus": bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'ngs'])] = o
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df.apply(calc_proportion)


def create_df_x(days, obs, name_part, bact_name, stds=0):
    data = {"Genus": bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'realx'])] = o
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df

# Save full ODE solutions (States included)
def create_df_fullx(days, obs, name_part, bact_name, stds=0):
    n_cl = len(bact_name)
    n_states = len(obs) // len(bact_name)
    data = {"Genus": [bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]}#bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'realx'])] = o[:2*n_cl]
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    return df

# Useful functions to work with dataframes
def preprocess_dataframe(df, cutoff=0.0, cutoff_prop=0., calc_prop=True):
    df = df[-np.all(df==0, axis=1)] # leave only columnes where bacteria is present

    # Drop bacteria that appear less then cutoff % in each measurement
    bact_abovecutoff = df.T[df.T>=cutoff].dropna(axis=1, how='all').columns

    df = df.T[bact_abovecutoff].T
    #df = df.apply(calc_proportion)
    # Drop bacteria that contribute less then 1 % to data
    df = df[df.T.sum()/df.T.sum().sum()>cutoff_prop] # only bacteria that has at least 'cutoff' presence in dataframe
    if calc_prop:
        return df.apply(calc_proportion), df.T.columns # bacteria names
    else:
        return df, df.T.columns


def drop_column(df, regs):
    return df[df.columns.drop(list(sum([df.filter(regex=reg)for reg in regs])))]


def calc_proportion(x):
    return x.astype(float)/ x.sum()#.astype(float)


def get_cluster_dataframe(df, clusters):
    d = {"Clusters": [f'Bacteria_{i:02d}' for i in range(len(clusters))]}
    for f in df.columns:
        d[f] = [df.T[cl].T.sum()[f] for cl in clusters]
    return pd.DataFrame(data=d).groupby('Clusters').sum()#.apply(calc_proportion) ## ???? (do we need to recalculate)


def merge_dfs(dfs):
    return pd.concat(dfs).groupby(level=0).sum()


def get_meas_days(df, exp):
    a = sorted(list(set([(f.split('_')[3]) for f in df.filter(like=exp).columns])))
    return np.asarray(a, dtype=float)


def filter_dataframe(exp, dfs):
    return [df.filter(like=exp) for df in dfs]

def filter_dataframe_regex(regex, dfs):
    return [df.filter(regex=regex) for df in dfs]

def update_df(df, func, *args):
    data = {}
    for f in df.columns:
        data[f] = func(df[f], f, *args)
    data[df.T.columns.name] = df.T.columns[:len(data[f])]
    df_new = pd.DataFrame(data=data).groupby(df.T.columns.name).sum()
    return df_new


def restore_missing_rows(df, clmns):
    df_new = df.T
    for i, b in enumerate(clmns):
        if not np.any(df_new.columns == b):
            #df_new.insert(i, b, [0. for _ in range (np.shape(df_new)[0])])
            df_new[b] = [0. for _ in range (np.shape(df_new)[0])]
    return df_new.T.sort_values(by=['Genus'], ascending=True)


def make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0.):
    df_ngs_red, bact_ngs = preprocess_dataframe(df_ngs, cutoff=cutoff)
    df_maldi_red, bact_maldi = preprocess_dataframe(df_maldi, cutoff=cutoff)

    bact_all = sorted(set(list(bact_maldi) + list(bact_ngs)))
    ind_ngs = [bact_all.index(i) for i in list(bact_ngs)]
    ind_maldi = [bact_all.index(i) for i in list(bact_maldi)]
    ind_not_in_maldi = [bact_all.index(i) for i in rest(bact_maldi, bact_ngs)]
    T_x = [1. if i in ind_ngs else 0. for i in range (len(bact_all))]
    df_ngs = restore_missing_rows(df_ngs, bact_all)
    df_maldi = restore_missing_rows(df_maldi, bact_all)

    s_x_predefined =  np.array([np.nan if i in ind_maldi else 0. for i in range (len(bact_all))])
    s_x_predefined[ind_not_in_maldi] = 1.
    media = ['MRS', 'PC']
    s_x_predefined_fin = np.zeros((len(media), len(s_x_predefined)))
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined_fin[i] = np.copy(s_x_predefined)
        s_x_predefined_fin[i, ind_med_null] = 0.
    return df_maldi, df_ngs, T_x, s_x_predefined_fin


def extract_observables_from_df_maldi(df_maldi, days, exps, media):
    obs_maldi = np.zeros((len(exps), len(media), np.shape(df_maldi)[0], len(days)))
    for i, exp in enumerate(exps):
        for j, med in enumerate(media):
            for k, d in enumerate(days):
                df0 = df_maldi.filter(like=exp).filter(like=med).filter(like=f'_{int(d):02d}_')
                if np.shape(df0)[-1] != 0.:
                    obs_maldi[i, j, :, k] = np.array(df0.T)[0]
                else:
                    obs_maldi[i, j, :, k] = np.nan*np.ones((np.shape(df_maldi)[0]))
    return obs_maldi
    

def extract_observables_from_df_ngs(df_ngs, days, exps):
    obs_ngs = np.zeros((len(exps), np.shape(df_ngs)[0], len(days)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days):
            df0 = df_ngs.filter(like=exp).filter(like=f'_{int(d):02d}_')
            if np.shape(df0)[-1] != 0.:
                obs_ngs[i, :, k] = np.array(df0.T)[0]
            else:
                obs_ngs[i, :, k] = np.nan*np.ones((np.shape(df_ngs)[0]))
    return obs_ngs
    

def extract_observables_from_df_mibi(df_mibi, days, exps, media):
    obs_mibi = np.zeros((len(exps), len(media), len(days)))
    std_mibi = np.zeros((len(exps), len(media), len(days)))
    for i, exp in enumerate(exps):
        for j, med in enumerate(media):
            for k, d in enumerate(days):
                df0 = df_mibi.filter(like=exp+'_').filter(like=med).filter(like=f'_{int(d):02d}_')
                if np.shape(df0)[-1] != 0.:
                    obs_mibi[i, j, k] = np.mean(df0.T['Average'])
                    std_mibi[i, j, k] = float(df0.T['Standard deviation'].values)
                else:
                    obs_mibi[i, j, k], std_mibi[i, j, k] = np.nan, np.nan
    return obs_mibi, std_mibi


def extract_observables_from_df(dfs):
    (df_mibi, df_maldi, df_ngs, ) = dfs
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns]))) # ['MRS', 'PC']

    days_maldi = sorted(set([float(f.split('_')[3]) for f in df_maldi.columns]))
    days_ngs = sorted(set([float(f.split('_')[3]) for f in df_ngs.columns]))
    days_mibi = sorted(set([float(f.split('_')[3]) for f in df_mibi.columns]))
    days_total = sorted(set(days_mibi + days_maldi + days_ngs))

    mibi = extract_observables_from_df_mibi(df_mibi, days_total, exps, media)
    maldi = extract_observables_from_df_maldi(df_maldi, days_total, exps, media)
    ngs = extract_observables_from_df_ngs(df_ngs, days_total, exps)
    return days_total , [mibi[0], maldi, ngs]

def add_rest_bacteria(df, bact_all):
    df0 = df.T
    df0['Others'] = 1-df.sum()
    return  df0.T, bact_all+['Others']

def get_values_from_dataframe(df, cutoff=0.):
    sample_names = df.columns
    days = [float(f.split('_')[3]) for f in sample_names]
    bacteria = df.T.columns
    observable = np.array([df[f].values for f in sample_names]).T
    #bacteria, observable = zip(*[[b, o] for b, o in sorted(zip(bacteria, observable), key=lambda x: -np.sum(x[1])) if np.sum(o)>cutoff])
    return days, observable, bacteria


def get_values_from_dataframe_biochem(df, meas):
    sample_names = df.columns
    days = [float(f.split('_')[3]) for f in sample_names]
    measurement = df.T.columns
    observable = np.array([df[f].values for f in sample_names]).T
    for m, o in zip(measurement, observable):
        if m == meas:
            obs = o        
            days, obs = zip(*[[d, o] for d, o in sorted(zip(days, obs), key=lambda x: x[0])])
    return days, obs
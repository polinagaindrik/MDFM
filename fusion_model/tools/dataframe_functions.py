import pandas as pd
import numpy as np


def intersection(lst1, lst2):
    """Return the values from the first list that also occur in the second."""
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
    

def rest(lst1, lst2):
    """Return the values from the first list that do not occur in the second."""
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


# Create dataframes for lnown days and observables
def create_df_mibi(days, obs, name_part, bact_name, media=['MRS', 'PC'], stds=None):
    """Build a MiBi dataframe with means, standard deviations, and replicate values."""
    if stds is None:
        stds = np.zeros(np.shape(obs))
    data = {'Properties': ['Average', 'Standard deviation', 'Replicas']}
    for d, o, std in zip(days, obs.T, stds.T):
        for med, o_med, std_med in zip(media, o, std):
            data["_".join(name_part + [f'{int(d):02d}', f'{med}-mibi'])] = [o_med, std_med, [o_med]]
    return pd.DataFrame(data=data).groupby('Properties').sum()


def create_df_sumx(days, obs, name_part, bact_name, stds=None):
    """Build a dataframe of total bacterial counts indexed by sample and time."""
    data = {'Properties': ['Average']}
    for d, o in zip(days, obs):   
        data["_".join(name_part + [f'{int(d):02d}', '-sumx'])] = [o]
    return pd.DataFrame(data=data).groupby('Properties').sum()


def create_df_maldi(days, obs, name_part, bact_name, media=['MRS', 'PC'], stds=0):
    """Build a genus-by-sample MALDI dataframe and normalize each sample to proportions."""
    data = {"Genus": bact_name}
    for d, o in zip(days, obs.T):
        for med, o_med in zip(media, o.T):
            data["_".join(name_part + [f'{int(d):02d}', f'{med}-maldi'])] = o_med
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df.apply(calc_proportion)


def create_df_ngs(days, obs, name_part, bact_name, **kwargs):
    """Build a genus-by-sample NGS dataframe and normalize each sample to proportions."""
    data = {"Genus": bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'ngs'])] = o
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df.apply(calc_proportion)


def create_df_x(days, obs, name_part, bact_name, **kwargs):
    """Build a genus-by-sample dataframe of modeled bacterial counts."""
    data = {"Genus": bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'realx'])] = o
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    #df[df < 0] = 0
    return df

# Save full ODE solutions (States included)
def create_df_fullx(days, obs, name_part, bact_name, stds=0):
    """Build a dataframe containing each cluster's separate ODE state values."""
    n_cl = len(bact_name)
    n_states = len(obs) // len(bact_name)
    data = {"Genus": [bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]}#bact_name}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'realx'])] = o[:2*n_cl]
    df = pd.DataFrame(data=data).groupby('Genus').sum()
    return df

# Useful functions to work with dataframes
def preprocess_dataframe(df, cutoff=0.0, cutoff_prop=0., calc_prop=True):
    """Remove absent and low-abundance bacteria, optionally normalizing each sample."""
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
    """Return a dataframe with columns matching any supplied regular expression removed."""
    return df[df.columns.drop(list(sum([df.filter(regex=reg)for reg in regs])))]


def calc_proportion(x):
    """Normalize values by their sum to produce proportions."""
    return x.astype(float)/ x.sum()#.astype(float)


def get_cluster_dataframe(df, clusters):
    """Aggregate rows of a dataframe into the specified clusters."""
    d = {"Clusters": [f'Bacteria_{i:02d}' for i in range(len(clusters))]}
    for f in df.columns:
        d[f] = [df.T[cl].T.sum()[f] for cl in clusters]
    return pd.DataFrame(data=d).groupby('Clusters').sum()#.apply(calc_proportion) ## ???? (do we need to recalculate)


def merge_dfs(dfs, sort=True):
    """Concatenate dataframes and sum rows that share an index."""
    return pd.concat(dfs).groupby(level=0, sort=sort).sum(min_count=1)


def get_meas_days(df, exp):
    """Extract sorted measurement days for an experiment from dataframe columns."""
    a = sorted(list(set([(f.split('_')[3]) for f in df.filter(like=exp).columns])))
    return np.asarray(a, dtype=float)


def filter_dataframe(exp, dfs):
    """Select columns containing an experiment identifier from each dataframe."""
    return [df.filter(like=exp) for df in dfs]

def filter_dataframe_regex(regex, dfs):
    """Select columns matching a regular expression from each dataframe."""
    return [df.filter(regex=regex) for df in dfs]

def update_df(df, func, *args):
    """Transform each dataframe column with a callback and regroup the results."""
    data = {}
    for f in df.columns:
        data[f] = func(df[f], f, *args)
    data[df.T.columns.name] = df.T.columns[:len(data[f])]
    df_new = pd.DataFrame(data=data).groupby(df.T.columns.name).sum()
    return df_new


def restore_missing_rows(df, clmns):
    """Add zero-valued rows for missing labels and sort rows by genus."""
    df_new = df.T
    for i, b in enumerate(clmns):
        if not np.any(df_new.columns == b):
            #df_new.insert(i, b, [0. for _ in range (np.shape(df_new)[0])])
            df_new[b] = [0. for _ in range (np.shape(df_new)[0])]
    return df_new.T.sort_values(by=['Genus'], ascending=True)


def make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0., media=['MRS', 'PC']):
    """Align MALDI and NGS genus rows and create their transfer and filter weights."""
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
    s_x_predefined_fin = np.zeros((len(media), len(s_x_predefined)))
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined_fin[i] = np.copy(s_x_predefined)
        s_x_predefined_fin[i, ind_med_null] = 0.
    return df_maldi, df_ngs, T_x, s_x_predefined_fin


def extract_observables_from_df_maldi(df_maldi, days, exps, media):
    """Extract MALDI measurements into experiment-by-medium-by-genus-by-day arrays."""
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
    """Extract NGS measurements into experiment-by-genus-by-day arrays."""
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
    """Extract MiBi means and standard deviations into experiment-by-medium arrays."""
    obs_mibi = np.zeros((len(exps), len(media), len(days)))
    std_mibi = np.zeros((len(exps), len(media), len(days)))
    for i, exp in enumerate(exps):
        for j, med in enumerate(media):
            for k, d in enumerate(days):
                df0 = df_mibi.filter(like=exp+'_').filter(like=med).filter(like=f'_{int(d):02d}_')
                if np.shape(df0)[-1] != 0.:
                    obs_mibi[i, j, k] = np.mean(df0.T['Average'])
                    # np.mean (like 'Average' above): float() on a 1-element array fails in numpy >= 2.x
                    std_mibi[i, j, k] = np.mean(df0.T['Standard deviation'].astype(float))
                else:
                    obs_mibi[i, j, k], std_mibi[i, j, k] = np.nan, np.nan
    return obs_mibi, std_mibi


def extract_observables_from_df_x(df_x, days, exps, media=None):
    """Extract modeled bacterial counts into experiment-by-genus-by-day arrays."""
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days):
            df0 = df_x.filter(like=exp).filter(like=f'_{int(d):02d}_')
            if np.shape(df0)[-1] != 0.:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan*np.ones((np.shape(df_x)[0]))
    return obs_x


def extract_observables_from_df(dfs):
    """Extract all observation types and return their shared day grid and data arrays."""
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
    """Add an Others row for abundance not represented by the listed bacteria."""
    df0 = df.T
    df0['Others'] = 1-df.sum()
    return  df0.T, bact_all+['Others']

def get_values_from_dataframe(df, cutoff=0.):
    """Return sample days, observation values, and row labels from a dataframe."""
    sample_names = df.columns
    days = [float(f.split('_')[3]) for f in sample_names]
    bacteria = df.T.columns
    observable = np.array([df[f].values for f in sample_names]).T
    #bacteria, observable = zip(*[[b, o] for b, o in sorted(zip(bacteria, observable), key=lambda x: -np.sum(x[1])) if np.sum(o)>cutoff])
    return days, observable, bacteria


def get_values_from_dataframe_biochem(df, meas):
    """Return sorted measurement days and values for one biochemical measurement."""
    sample_names = df.columns
    days = [float(f.split('_')[3]) for f in sample_names]
    measurement = df.T.columns
    observable = np.array([df[f].values for f in sample_names]).T
    for m, o in zip(measurement, observable):
        if m == meas:
            obs = o        
            days, obs = zip(*[[d, o] for d, o in sorted(zip(days, obs), key=lambda x: x[0])])
    return days, obs
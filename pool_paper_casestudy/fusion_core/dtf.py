"""
Dataframe utilities.

``merge_dfs`` is ported from ``fusion_model/tools/dataframe_functions.py``.

``extract_observables_from_df`` was previously duplicated inside
``pool_paper_casestudy/pool_model_functions.py``. It is a generic dataframe
utility -- it only relies on the ``{experiment}_..._{day:02d}_..."`` wide
column-naming convention used by every dataframe in this case study, and
does not reference any bacteria names, ODE model, or other pool-paper-
specific detail -- so it belongs here as shared/common code rather than
being redefined case-study by case-study, next to the fusion_model original
``extract_observables_from_df_x`` that plays the same role for a single
dataframe.
"""

import numpy as np
import pandas as pd


def merge_dfs(dfs, sort=True):
    """Concatenate a list of dataframes and sum duplicate (index) rows."""
    return pd.concat(dfs).groupby(level=0, sort=sort).sum()


def create_df_poolpaper(days, obs, name_part, bact_name, stds=0):
    """Build a pool-paper-format wide dataframe (moved from
    `pool_paper_casestudy/pool_model_functions.py`): one row per measured
    quantity (`x_<bacterium>_State_00`, `m_BAC`, `m_LA`, `pH`), one column
    per `{name_part}_{day:02d}_poolpaper`.
    """
    n_cl = len(bact_name)
    n_states = 1
    data = {"Measurement": ['x_'+bact_name[i]+f'_State_{j:02d}' for i in range (n_cl) for j in range (n_states)]+['m_BAC', 'm_LA', 'pH']}
    for d, o in zip(days, obs.T):
        data["_".join(name_part + [f'{int(d):02d}', 'poolpaper'])] = o
    df = pd.DataFrame(data=data).set_index('Measurement')
    #df = pd.DataFrame(data=data).groupby('Measurement', sort=False).sum()
    return df


def extract_observables_from_df(dfs):
    """Turn a single wide dataframe `dfs = [df_x]` into a (days, [obs_x]) array.

    `df_x` columns are expected to follow the
    `{experiment}_..._{day:02d}_{...}` naming convention; rows are the
    measured quantities (its own index). Missing (experiment, day)
    combinations are filled with NaN.
    """
    (df_x,) = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split("_")[-2]) for f in df_x.columns]))
    obs_x = np.zeros((len(exps), np.shape(df_x)[0], len(days_x)))
    for i, exp in enumerate(exps):
        for k, d in enumerate(days_x):
            df0 = df_x.filter(like=exp).filter(like=f"_{int(d):02d}_")
            if np.shape(df0)[-1] != 0.0:
                obs_x[i, :, k] = np.array(df0.T)[0]
            else:
                obs_x[i, :, k] = np.nan * np.ones((np.shape(df_x)[0]))
    return days_x, [obs_x]

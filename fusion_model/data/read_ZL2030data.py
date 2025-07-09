#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pybis import Openbis
import getpass

from ..tools.dataframe_functions import drop_column, merge_dfs, calc_proportion
from fusion_model.tools import dataframe_functions as dtf
from .output import read_from_json


def get_bacteria_for_model(exps_maldi, exps_ngs, path_data, cutoff=0., cutoff_prop=0.,):
    df_ngs = get_all_experiments_dataframe(read_ngs, exps_ngs, path_data+'NGS/')
    df_ngs = dtf.drop_column(df_ngs, ['M2'])
    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)

    df_maldi = get_all_experiments_dataframe(read_maldi, exps_maldi, path_data+'MALDI/')
    df_maldi = dtf.drop_column(df_maldi, ['M2', 'VRBD'])
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    bact_all = dtf.intersection(bact_ngs.values, bact_maldi.values)
    return bact_all, df_maldi.T[bact_all].T, df_ngs.T[bact_all].T


def read_all_data_in_df(exps_mibi, exps_maldi, exps_ngs, path_data, path_out, bact_all=None, cutoff=0., cutoff_prop=0., add_name=''):
    df_mibi = get_all_experiments_dataframe(read_mibi, exps_mibi, path_data+'microbiology/')
    df_mibi =dtf.drop_column(df_mibi, ['M2', 'VRBD'])
    if bact_all is None:
        df_maldi = get_all_experiments_dataframe(read_maldi, exps_maldi, path_data+'MALDI/')
        df_ngs = get_all_experiments_dataframe(read_ngs, exps_ngs, path_data+'NGS/')
        # Restore ngs and maldi dataframes on missing bacteria
        #bact_all = sorted(set(list(bact_ngs.values) + list(bact_maldi.values)))
        #df_maldi, df_ngs, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=cutoff0)
        df_ngs = df_ngs.T[bact_all].T
        df_maldi = df_maldi.T[bact_all].T
    else:
        bact_all, df_maldi, df_ngs = get_bacteria_for_model(exps_maldi, exps_ngs, path_data, cutoff=cutoff, cutoff_prop=cutoff_prop)

    df_maldi, bact_all = dtf.add_rest_bacteria(df_maldi, bact_all)
    df_ngs, _ = dtf.add_rest_bacteria(df_ngs, bact_all)

    T_x = np.ones((len(bact_all)))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    s_x_predefined = np.ones((len(media), len(bact_all)))*np.nan
    for i, med in enumerate(media):
        bact_med_null = df_maldi.filter(like=med)[df_maldi.filter(like=med).T.sum()==0].T.columns
        ind_med_null = [bact_all.index(i) for i in list(bact_med_null)]
        s_x_predefined[i, ind_med_null] = 0.

    data = [df_mibi, df_maldi, df_ngs]
    df_mibi.to_pickle(path_out+f'dataframe_mibi{add_name}.pkl')
    df_maldi.to_pickle(path_out+f'dataframe_maldi{add_name}.pkl') 
    df_ngs.to_pickle(path_out+f'dataframe_ngs{add_name}.pkl')

    df_mibi.to_csv(path_out+f'dataframe_mibi{add_name}.csv', index=False)
    df_maldi.to_csv(path_out+f'dataframe_maldi{add_name}.csv')
    df_ngs.to_csv(path_out+f'dataframe_ngs{add_name}.csv')

    print(bact_all)
    data = dtf.filter_dataframe_regex('V.._', data)
    return data, T_x, s_x_predefined


def prepare_ZL2030_data(exps_mibi, exps_maldi, exps_ngs, cutoff=0., cutoff_prop=0., path='', add_name=''):
    bact_all, df_maldi_all, df_ngs_all = get_bacteria_for_model(exps_maldi, exps_ngs, 'experiments/', cutoff=cutoff, cutoff_prop=cutoff_prop)
    dfs_calibr, T_x, s_x_predefined = read_all_data_in_df(exps_mibi, exps_maldi, exps_ngs, 'experiments/', path, bact_all=bact_all,
                                                          cutoff=cutoff, cutoff_prop=cutoff_prop, add_name=add_name)
    return dfs_calibr, bact_all, T_x, s_x_predefined


def prepare_ZL2030_data_for_prediction(exps, bact_all, path_data, path_out='', add_name=''):
    df_mibi = get_all_experiments_dataframe(read_mibi, exps, path_data+'microbiology/')
    df_mibi = dtf.drop_column(df_mibi, ['M2', 'VRBD'])
    df_maldi = get_all_experiments_dataframe(read_maldi, exps, path_data+'MALDI/')
    df_maldi = dtf.drop_column(df_maldi, ['M2', 'VRBD'])
    df_maldi, _ = dtf.preprocess_dataframe(df_maldi, cutoff=0., cutoff_prop=0., calc_prop=False)

    df_ngs = get_all_experiments_dataframe(read_ngs, exps, path_data+'NGS/')
    df_ngs = dtf.drop_column(df_ngs, ['M2'])
    df_ngs, _ = dtf.preprocess_dataframe(df_ngs, cutoff=0., cutoff_prop=0., calc_prop=False)

    df_maldi = dtf.restore_missing_rows(df_maldi, bact_all)
    df_ngs = dtf.restore_missing_rows(df_ngs, bact_all)
    df_ngs = df_ngs.T[bact_all].T
    df_maldi = df_maldi.T[bact_all].T
    #df_maldi, bact_all = add_rest_bacteria(df_maldi, bact_all)
    #df_ngs, bact_all = add_rest_bacteria(df_ngs, bact_all)

    for df, meas in zip([df_mibi, df_maldi, df_ngs], ['mibi', 'maldi', 'ngs']):
        df.to_pickle(path_out+f'dataframe_{meas}{add_name}.pkl')
    df_mibi.to_csv(path_out+f'dataframe_mibi{add_name}.csv', index=False)
    df_maldi.to_csv(path_out+f'dataframe_maldi{add_name}.csv')
    df_ngs.to_csv(path_out+f'dataframe_ngs{add_name}.csv')
    return [df_mibi, df_maldi, df_ngs]


def _get_exp_filename(exp_name):
    # Redefine experiment names to read data
    if exp_name[0] == 'V':
        exp_filename = 'Experiment_' + exp_name[1:]
    elif exp_name[0] == 'P':
        exp_filename = 'Pilot_' + exp_name[1:]
    return exp_filename


def _correct_df_for_tempinterruption(df):
    names_init = set([col.split('_')[0] for col in df.columns])
    for Tshift in ['CCD01', 'CCD02', 'RP']:
        if np.size((df.filter(regex=Tshift))) != 0:
            for day in ['_00_', '_01_']:
                for f in df.filter(like=day).columns:
                    fspl = f.split('_')
                    fspl[0] = '-'.join([fspl[0], Tshift])
                    df["_".join(fspl)] = df[f]
    names_fin = set([col.split('_')[0] for col in df.filter(like='_00_').filter(like='PC').columns])
    df = drop_column(df, [' ']+[value+'_' for value in names_fin if value not in names_init])
    return df


def read_mibi(exp_name, dir_data=''):
    exp_temps = read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    exp_filename = _get_exp_filename(exp_name)
    # Read the file:
    filename = f'Data on colony counts - {exp_filename}.xlsx'
    if int(exp_name.split('_')[0][1:]) >= 14:
        df0 = pd.read_excel(dir_data + filename, keep_default_na=True, usecols='A:G')
    else:
        df0 = pd.read_excel(dir_data + filename, keep_default_na=True) # , on_bad_lines='skip'
    df0 = df0.dropna(axis=1, how='all')[-np.isnan(df0.Average)]

    d = {'Properties': ['Average', 'Standard deviation', 'Replicas']}
    for f in df0[df0.columns[0]].values:
        fspl = f.split('_')
        if len(fspl) == 8:
            f1 = fspl[0]
        elif len(fspl) == 9:
            f1 = "-".join([fspl[0], fspl[3]])
        elif len(fspl) == 10:
            if fspl[4] == 'REPACK':
                fspl[4] = 'RP'
            f1 = "-".join([fspl[0], fspl[3], fspl[4]])
        elif len(fspl) == 11:            
            if fspl[4] == 'REPACK':
                f1 = "-".join([fspl[0], fspl[3], fspl[5], 'RP'])
            else:
                f1 = "-".join([fspl[0], fspl[3], fspl[4], fspl[5]])            
        df_select = df0[df0[df0.columns[0]] == f].iloc[:, 2:]
        df_select = df_select.values[~np.isnan(df_select.values)]

        if fspl[-4] != '00':
            d["-".join(["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", fspl[-4], fspl[-1][:-2]]), 'mibi'])] = [
                np.mean(df_select),
                np.std(df_select),
                df_select
            ]
        else:
            for media in ['M1', 'M2']:
                d["-".join(["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", fspl[-4], fspl[-1][:-2]]), 'mibi'])] = [
                    np.mean(df_select),
                    np.std(df_select),
                    df_select
                ]
    df = _correct_df_for_tempinterruption(pd.DataFrame(data=d))
    return df.groupby('Properties').sum()


def read_maldi(exp_name, dir_data=''):
    filename =  f'Data on bacteria identified by MALDI-Biotyper - {_get_exp_filename(exp_name)}'
    media_filenames = ['_MRS_Aerob_wachsende_Milchsäurebakterien', '_PC_Aerobe_Keimzahl (Gesamtkeimzahl)', '_VRBD_Enterobacterales']
    if int(exp_name[1:3]) < 15:
        df_maldi = pd.read_excel(dir_data + filename+ '.xlsx', keep_default_na=True)
        df_maldi =  _df_maldi_transform(df_maldi, exp_name, dir_data)
    elif int(exp_name[1:3]) == 15:
        df_maldi = pd.read_excel(dir_data + filename + '.xlsx', keep_default_na=True, skiprows=1)
        df_maldi = _df_maldi_transform(df_maldi, exp_name, dir_data)
    elif int(exp_name[1:3]) >= 16 and int(exp_name[1:3])<=19:
        df_maldi = []
        for media in media_filenames:
            df = pd.read_excel(dir_data + filename+media + '.xlsx', keep_default_na=True)
            df_maldi.append(_df_maldi_transform(df, exp_name, dir_data))
        df_maldi = merge_dfs(df_maldi)
    else:
        df_maldi = pd.read_excel(dir_data + filename+media_filenames[1] + '.xlsx', keep_default_na=True)
        df_maldi = _df_maldi_transform(df_maldi, exp_name, dir_data)
    return df_maldi
        

def _df_maldi_transform(df, exp_name, dir_data=''):
    df=df.rename(columns={"Unnamed: 0": "Species", "Taxa": "Species"})
    df=df.sort_values(by = "Species").reset_index(drop=True)
    df = df[df['Species'].isna()==False]  # noqa: E712
    df.insert(0, "Genus", [sp.strip().split()[0] for sp in df["Species"].values])
    df = df.dropna(axis=1, how='all').drop(columns=['Species'])
    df['Genus'] = ['Leuconostoc' if f=='Leucononstoc' else f for f in df['Genus'].values]
    #Leucononstoc carnosum_MRI_FLM270
    df = df.groupby('Genus').sum()
    # Get undentified isolates:
    #df_unid = read_pkl('MALDI_unidentified.pkl', dir_data).filter(like=exp_name)
    # for now drop Experiments 10-16
    #df = merge_dfs([df, df_unid])

    # Rename samples
    for f in df.__iter__():
        df = df.rename(columns={f:_rename_maldi_samples(f)})
    df = df.T.groupby(by=df.columns).sum().T

    # Get M1/M2 indentificators instead of M0
    for f in df.columns:
        if f.split('_')[3] == '00':
            df["_".join([ff if i!=1 else 'M1' for i, ff in enumerate(f.split('_'))])] = df[f].values
            df = df.rename(columns={f:"_".join([ff if i!=1 else 'M2' for i, ff in enumerate(f.split('_'))])})
    df = _correct_df_for_tempinterruption(df)
    return df.apply(calc_proportion)


def _rename_maldi_samples(f):
    fspl = f.split('_')
    if fspl[-1] == 'IDENExp01':
        fspl = fspl[:-1]
    if len(fspl) == 8:
        return  "-".join(["_".join([fspl[0], fspl[2], fspl[-5], fspl[-4], fspl[-1][:-2]]), 'maldi'])
    elif len(fspl) == 9:
        f1 = "-".join([fspl[0], fspl[3]])
    elif len(fspl) == 10:
        f1 = "-".join([fspl[0], fspl[3], fspl[4]])
    elif len(fspl) == 11:
        f1 = "-".join([fspl[0], fspl[3], fspl[4], fspl[5]])
    return "-".join(["_".join([f1, fspl[2], fspl[-5], fspl[-4], fspl[-1][:-2]]), 'maldi'])


def read_ngs(exp_name, **kwargs):
    if int(exp_name[1:]) <= 7:
        return read_ngs1(exp_name, **kwargs)
    else:
        return read_ngs2(exp_name, **kwargs)


def read_ngs1(exp_name, dir_data=''):
    exp_temps = read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    temp = exp_temps[exp_name[:3]]
    filename = f'ZOTU_Table_{exp_name}.tsv'
    df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3], keep_default_na=True, encoding="ISO-8859-1")
    df = df.dropna(axis=1, how='all')
    df=df.rename(columns={"70% N2 + 30% CO2": "M2", "70% O2 + 30% CO2": "M1"})
    d = {"Genus": ["_".join(bact.split(";")[-2:-1]) if bact.split(";")[0]!='Eukaryota' else '' for bact in df[df.columns[1]].values]}
    for f in df.filter(like='M').columns:
        d["_".join([exp_name, f[0], f'{int(temp):02d}C', f'{int(f[1]):02d}', 'ngs'])] = df[f].values
    df = pd.DataFrame(data=d)
    df = df[df.Genus!='']
    return df.groupby('Genus').sum().apply(calc_proportion)


def read_ngs2(exp_name, dir_data=''):
    filename = f'ZOTU_Table_{exp_name.split("-")[0]}.tsv'
    if int(exp_name[1:])<12 and int(exp_name[1:])>=8:
        df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3, 4], skiprows=[5, 6],  keep_default_na=True, encoding="ISO-8859-1")
    elif int(exp_name[1:])>=12:
        df = pd.read_csv(dir_data + filename, sep='\t', header=[0, 1, 2, 3, 4, 5, 6], skiprows=[7, 8],  keep_default_na=True, encoding="ISO-8859-1")

    df = df.drop(columns=[df.columns[1]]).dropna(axis=1, how='all')
    d = {"Genus": ["_".join(bact.split(";")[-2:-1]) if bact.split(";")[0]!='Eukaryota' else ''
                   for bact in df[df.columns[0]].values]}
    for f in df.filter(like='V').__iter__():
        if f[2].strip() == 'O2':
            atm = 'M1'
        elif f[2].strip() == 'N2':
            atm = 'M2'
        if f[4].strip()=='6h':
            exp_name_new = "-".join([exp_name, 'CCD01'])
        elif f[4].strip()=='12h':
            exp_name_new = "-".join([exp_name, 'CCD02'])
        elif f[-2].strip()=='SHL1' or f[-2].strip()=='SHL2':
            exp_name_new = "-".join([exp_name, f[-2].strip()])
        elif f[-1].strip() == '7d':
            exp_name_new = "-".join([exp_name, 'STOR01'])
        else:
            exp_name_new = exp_name    
        d["_".join([exp_name_new, atm, f'{int(f[1].strip()[0]):02d}C', f[3][1:], 'ngs'])] = df[f].values
    df = pd.DataFrame(data=d)
    df = df[df.Genus!='']
    df = _correct_df_for_tempinterruption(df)
    return df.groupby('Genus').sum().apply(calc_proportion)


def read_biochem(folder_path, username, password=None):
    o = login_openbis(username, password)
    d = {'Measurement': ['O2', 'CO2']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*GAS*'):
        d[s.props.sample_name] = [float(s.props.gas_measurement_oxygen), float(s.props.gas_measurement_carbondioxid)]
    df_gas =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_gas = _rename_df_biochem(df_gas)
    df_gas.to_pickle(folder_path+'biochem_o2_co2.pkl')

    d = {'Measurement': ['TBARS', 'POZ']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*TBARS_POZ*'):
        if s.props.tbars_measurement_mean is not None:
            d[s.props.sample_name] = [float(s.props.tbars_measurement_mean), float(s.props.poz_measurement_mean)]
    df_tb =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_tb = _rename_df_biochem(df_tb)

    d = {'Measurement': ['pH']}
    for s in o.get_samples(space='ZL2030', project='WORKPACKAGE_1.2', experiment='*PH_AW*'):
        if s.props.ph_measurement_mean is not None:
            d[s.props.sample_name] = [float(s.props.ph_measurement_mean)]
    df_ph =  pd.DataFrame(data=d).groupby('Measurement').mean()
    df_ph = _rename_df_biochem(df_ph)
    df = merge_dfs([df_tb, df_ph])
    df = df.replace(0, np.nan, inplace=False)
    df.to_pickle(folder_path+'biochem_oh_tbars_poz.pkl')
    return df, df_gas


def read_pkl(name, folder_path):
    return pd.read_pickle(folder_path+name)


def _rename_df_biochem(df):
    for f in df.columns:
        fspl = f.split('_')
        if len(fspl) == 6:
            new_name = "_".join([fspl[0], fspl[2], fspl[-3], fspl[-2], 'biochem'])
        elif len(fspl) == 7:
            f1 = f1 = "-".join([fspl[0], fspl[3]])
            new_name = "_".join([f1, fspl[2], fspl[-3], fspl[-2], 'biochem'])
        df = df.rename(columns={f:new_name})
    return df.T.groupby(by=df.columns).mean().T


def read_color(exp_name, dir_data=''):
    exp_temps = read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    filename = f'Data of color measurement main trial {int(exp_name[1:])} KI.xlsx'
    df0 = pd.read_excel(dir_data + filename, keep_default_na=True, header=[0])

    d = {'Properties': df0.columns[2:]}
    for f in df0[df0.columns[0]].values:
        fspl = f.split('_')
        if len(fspl) == 9 or (len(fspl) == 8 and fspl[3][-1] != 'C'):
            f1 = "-".join([fspl[0], fspl[3]])
            day = fspl[5]
        else:
            f1 = fspl[0]
            day = fspl[4]

        df_select = df0[df0[df0.columns[0]] == f].iloc[:, 2:].mean()
        df_select = df_select.values[~np.isnan(df_select.values)]
    
        if fspl[-2][:3] != 'MRI':
            meas_name = '-'.join([fspl[-2], 'CLR'])
        else:    
            meas_name = 'CLR'
        if fspl[-4] != '00':
            d["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", day, meas_name])] = df_select
        else:
            for media in ['M1', 'M2']:
                d["_".join([f1, fspl[2], f"{int(exp_temps[fspl[0][:3]]):02d}C", day, meas_name])] = df_select
    df = _correct_df_for_tempinterruption(pd.DataFrame(data=d))
    return df.groupby('Properties').mean()


def get_all_experiments_dataframe(read_func, exp_names, dir_data):
    return merge_dfs([read_func(exp, dir_data=dir_data) for exp in exp_names])


def login_openbis(username, password=None):
    o = Openbis('https://openbis.zl2030.de/openbis/webapp/eln-lims/')
    if password is None:
        password = getpass.getpass()
    o.login(username, password, save_token=True)
    return o


def download_from_openbis(o, folder_path, workpackage, experiment_name, type_data=None):
    for dataset in o.get_datasets(space='ZL2030', project=workpackage, experiment=experiment_name, type=type_data):
        dataset.download(destination=folder_path, create_default_folders=False)

def get_MALDI_metadata(o, folder_path, workpackage, experiment_name):
    d = {'Genus': 'unidentified'}
    for s in  o.get_samples(space='ZL2030', project=workpackage, experiment=experiment_name):#,
                            #where={'experimental_subsample_name':'*V0*'}): ## For now just exp till 9
        if s.props.maldi_isolates_unidentified is None:
            unid = 0.
        else:
            unid = float(s.props.maldi_isolates_unidentified)
        d[s.props.experimental_subsample_name] = [unid] #s.props.maldi_isolates_analysed
    df =  pd.DataFrame(data=d).groupby('Genus').sum()
    df.to_pickle(folder_path+'MALDI_unidentified.pkl')
        

# Download excels with microbiological data from Openbis
def download_microbiology_from_openbis(folder_path, username, password=None):
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*COLONY_COUNT*')
    o.logout()

def download_MALDI_from_openbis(folder_path, username, password=None):
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*MALDI_BIOTYPER*')
    get_MALDI_metadata(o, folder_path, 'WORKPACKAGE_1.2', '*MALDI_BIOTYPER*')
    o.logout()
    
def download_NGS_from_openbis(folder_path, username, password=None):
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.5', '*AMPLICON_SEQ*', 'PROCESSED_DATA')
    o.logout()

def download_colordata_from_openbis(folder_path, username, password=None):
    o = login_openbis(username, password)
    download_from_openbis(o, folder_path, 'WORKPACKAGE_1.2', '*COLOR*')
    o.logout()
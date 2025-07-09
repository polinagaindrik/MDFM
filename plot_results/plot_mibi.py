import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm


scatter_marker = ['^', 'o', 'D', "s", "X", ".", "*", "P", "d", "x", '^', 'o', 'D', "s",]
mrkrs = {'CCD01': '^', 'CCD02': 'o', '': "x", 'RP':'D'}
lsts = {'CCD01': 'solid', 'CCD02': 'solid', '': "dashed"}
#plt.rcParams['figure.dpi'] = 500


if __name__ == "__main__":
    output_path = 'out/plot_mibi/'
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    # Define model, method, data used for model calibration 
    #username = ''
    #password = ''
    #fm.data.download_microbiology_from_openbis('experiments/microbiology/', username, password)
    #fm.data.download_MALDI_from_openbis('experiments/MALDI/', username, password)
    #fm.data.download_NGS_from_openbis('experiments/NGS/', username, password)

    exps = [f'V{i:02d}' for i in range(1, 22)] + [f'V{i:02d}' for i in range(23, 29)]
    df_mibi = fm.data.get_all_experiments_dataframe(fm.data.read_mibi, exps, 'experiments/microbiology/')
    df_mibi.to_pickle('experiments/microbiology/'+'dataframe_mibi.pkl')
    df_mibi = fm.data.drop_column(df_mibi, ['M2', 'VRBD'])


    exps_full = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    df_mibi_constT = df_mibi.filter(regex='V.._')

    media = 'PC'
    df_mibi = df_mibi.filter(like=media)


    #exp_plot = ['V10-CCD02', 'V11-CCD02', 'V11', 'V17-CCD02', 'V17', 'V19-CCD02', 'V19']
    exp_plot = ['V10-CCD01', 'V12-CCD01', 'V12', 'V18-CCD01', 'V18', 'V20-CCD01', 'V20']
    fm.plotting.plot_measurements_ZL2030_Tunterbrech(df_mibi, exp_plot, dir=output_path, add_name=f'MiBi_{media}_Tunterbrechung_6St')


    df_mibi_constT = df_mibi.filter(regex='V.._')
    fm.plotting.plot_all([2., 10., 14.], ('day', 'log bacterian count'), templ_meas=fm.plotting.plot_measurements_ZL2030_consttemp, df=df_mibi_constT, #time_lim=[18., 11., 11],
              dir=output_path, add_name=f'MiBi_{media}_const_temp_PL')

    df_mibi_industry = df_mibi.filter(regex='SLH')
    exps_SLH = sorted(list(set([s.split('_')[0] for s in df_mibi_industry.columns])))
    exp_plot2 = ['V15-SLH01', 'V15-SLH02','V21-SLH03a', 'V21-SLH03b' ]#, 'V22-SLH01', 'V22-SLH02']
    fm.plotting.plot_measurements_industry(df_mibi, exps_SLH, dir=output_path, add_name=f'MiBi_{media}_industry')

    exp_plot3 =  ['V13', 'V13-STOR01', 'V14', 'V14-STOR01', 'V16', 'V16-STOR01']
    fm.plotting.plot_measurements_stored_meat(df_mibi, exp_plot3, dir=output_path, add_name=f'MiBi_{media}_stored_meat')
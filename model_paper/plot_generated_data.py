import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
import paper_paths as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    # Folder (see paper_paths.py): STUDY = 'model_complexity', 'noise_vs_nspecies' or 'media_influence' (MEDIA)
    STUDY = 'model_complexity'
    n_cl = 6
    n_media = 2
    relnoise = 0.1
    MEDIA = None  # media_influence only, e.g. 'gen2+sel2' (sets n_media)
    if MEDIA is not None:
        n_media = pp.media_n(MEDIA)
    path = pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=MEDIA)
    path2 = path
    path_datagen = pp.ensure_dir(pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=MEDIA,
                                               sub='data_generation'))
    add_name = pp.add_name(n_cl, n_media, media=MEDIA)

    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
    df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
    (df_mibi, df_maldi, df_ngs, ) = data

    # Get 'real' model parameters used for data generation
    setup_real = fm.data.read_from_json('Result_temp_together_real.json', dir=path2)
    param_ode = np.array(setup_real['param_ode'])
    T_x = np.array(setup_real['T_x'])
    s_x = np.array(setup_real['s_x']).reshape((n_media, -1))

    L_0 = np.array(fm.data.read_from_json(fm.data.SETUP_DIR + 'Initial_values_x0_paper.json')['x0'])[:len(exps), :n_cl]
    x_0 = fm.data.set_initial_vals(L_0, exps, n_cl)

    # Solve model ODE:
    calibr_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'output_path': path2,
        'exp_temps': fm.data.read_exp_temps(path2, data[0]),
        's_x': s_x,
        'media': media, 
    }
    t_model = np.linspace(0., 18., 100)
    x_count, obs_mibi_model, obs_maldi_model, obs_ngs_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    # Plotting
    exp_plot = exps#['V01', 'V04', 'V07']
    for exp in exp_plot:
        fm.plotting.plot_opt_res_realx(df_x.filter(like=exp), t_model, x_count[int(exp[1:])-1], exp, path=path_datagen+'Data_generation2_', add_name=add_name, clrs=clrs1, plot_meas=False)#, clrs=clrs, add_name=add_name)  
        #fm.plotting.plot_opt_res_ngs(df_ngs.filter(like=exp), t_model, obs_ngs_model[int(exp[1:])-1], exp, path=path_datagen+'Data_generation_', add_name=add_name, clrs=clrs1)
        #fm.plotting.plot_opt_res_mibi(df_mibi.filter(like=exp), t_model, obs_mibi_model[int(exp[1:])-1], exp, media=['sel1', 'gen1'], path=path_datagen+'Data_generation_', add_name=add_name, clrs=clrs1)
            
        # Check if results are correct here?
        #fm.plotting.plot_opt_res_maldi(df_maldi.filter(like=exp), t_model, obs_maldi_model[int(exp[1:])-1], exp, media=['sel1', 'gen1'], path=path_datagen+'Data_generation_', add_name=add_name, clrs=clrs1)
      
import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

if __name__ == "__main__":
    colors_all = {
        'grey': '#808080',
        'red':'#D06062',
        'blue1': '#4E89B1',
        'purple':'#7E57A5',
        'brown':'#99582A',
        'orange_light':'#c79758',
        'yellow':'#E2B100',
        'green_dark':'#386641',
        'blue_bright':'#0982A4',
        'green_light':'#679E48',
        'orange':'#ED733E',
        'pink':'#C3568A',
    }
    # Set common plotting parameters for all figures
    plt.rc('text', usetex=True)
    rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

    rcParams['lines.linewidth'] = 2.
    rcParams['lines.linestyle'] = 'dashed'#'solid' #
    rcParams['lines.markersize'] = 8
    rcParams['figure.figsize'] = (7, 5)
    rcParams['legend.framealpha'] = 0.
    rcParams['legend.handlelength'] = 2.

    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['axes.labelsize'] = 15
    rcParams['legend.fontsize'] = 15#13

    rcParams['figure.dpi'] = 500

    n_cl = [4, 6, 8, 10, 12]
    n_media = 2
    relnoise=0.1

    #x_real = []
    #x_model = []
    rms = []
    rms_mibi, rms_maldi, rms_ngs = [], [], []
    x_max_val = 1e10
    L_0_real = np.array(fm.data.read_from_json('model_paper/out/Initial_values_x0_paper.json')['x0'])
    for n in n_cl:
        path = f'model_paper/out/model_complexity/{int(n)}_dim_{n_media}media_exp_{int(relnoise*100)}noise/calibration/'
        path2 = f'model_paper/out/model_complexity/{int(n)}_dim_{n_media}media_exp_{int(relnoise*100)}noise/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media)}media'
        df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
        exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))

        # Get initial real x values from dataframe
        days_meas = [fm.dtf.get_meas_days(df_x, exp) for exp in exps]
        x_real = fm.dtf.extract_observables_from_df_x(df_x, days_meas[0], exps)
        days_total, [obs_mibi_data, obs_maldi_data, obs_ngs_data] =fm.dtf.extract_observables_from_df(data)

        # Get 'real' model parameters used for data generation
        setup_real = fm.data.read_from_json('Result_temp_together_real.json', dir=path2)
        param_ode_real = np.array(setup_real['param_ode'])
        T_x_real = np.array(setup_real['T_x'])
        s_x_real = np.array(setup_real['s_x']).reshape((n_media, -1))
        calibr_setup_real = {
            'model': fm.mdl.fusion_model2,
            'T_x': T_x_real,
            'output_path': path2,
            'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
            's_x': s_x_real,
            'media': media, 
        }
        t_model = np.linspace(0, 18, 100)
        x_real2, obs_mibi_real, obs_maldi_real, obs_ngs_real, temps_real = fm.mdl.calc_obs_model(data, param_ode_real, calibr_setup_real, t_model)

        # Calculate optimized x values
        optim_file2 = "optimization_history1.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        T_x = [1. for _ in range (n)]
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
        s_x = np.array(param_opt)[-n*n_media:].reshape((n_media, n))
        param_ode = param_opt[:-n*n_media]
        calibr_setup={
            'model': fm.mdl.fusion_model2,
            'T_x': T_x,
            'output_path': path2,
            'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
            's_x': s_x,
            'media': media, 
        }

        #x_count, obs_mibi_model, obs_maldi_model, obs_ngs_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, days_meas[0])
        x_count, obs_mibi_model, obs_maldi_model, obs_ngs_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)
        #rms.append(root_mean_squared_error(x_real.flatten()/x_max_val, x_count.flatten()/x_max_val))
        #rms_mibi.append(root_mean_squared_error(obs_mibi_model.flatten()/x_max_val, obs_mibi_data.flatten()/x_max_val))
        #rms_maldi.append(root_mean_squared_error(obs_maldi_model.flatten(), obs_maldi_data.flatten()))
        #rms_ngs.append(root_mean_squared_error(obs_ngs_model.flatten(), obs_ngs_data.flatten()))

        rms.append(root_mean_squared_error(np.log(x_real2.flatten()), np.log(x_count.flatten())))
        rms_mibi.append(root_mean_squared_error(np.log(obs_mibi_model.flatten()), np.log(obs_mibi_real.flatten())))
        rms_maldi.append(root_mean_squared_error(obs_maldi_model.flatten(), obs_maldi_real.flatten()))
        rms_ngs.append(root_mean_squared_error(obs_ngs_model.flatten(), obs_ngs_real.flatten()))

    '''
    # Now get result for 10 species for 1 different media:
    # general media
    n_media = 1
    rms_media = []
    media_names = ['gen', 'sel']
    for med in media_names:
        add_name = f'_{int(n_cl[-1])}dim_{int(n_media)}media_{med}'
        path = f'model_paper/out/{int(n_cl[-1])}_dim_{int(n_media)}media_{med}/calibration/'
        path2 = f'model_paper/out/{int(n_cl[-1])}_dim_{int(n_media)}media_{med}/calibration/'
        df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
        exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))
        days_meas = [fm.dtf.get_meas_days(df_x, exp) for exp in exps]
        x_real = fm.dtf.extract_observables_from_df_x(df_x, days_meas[0], exps)
        optim_file2 = "optimization_history1.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
        s_x = np.array(param_opt)[-n*n_media:].reshape((n_media, n))
        param_ode = param_opt[:-n*n_media]
        calibr_setup={
                'model': fm.mdl.fusion_model2,
                'T_x': T_x,
                'output_path': path2,
                'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
                's_x': s_x,
                'media': media, 
        }
        x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, days_meas[0])
        rms_media.append(root_mean_squared_error(x_real.flatten()/x_max_val, x_count.flatten()/x_max_val))
    '''

    fig, ax = plt.subplots()
    lines = []
    labels = [r'log $x(t)$', r'log Plate Count', 'MALDI', 'NGS']
    clrs = [colors_all['blue1'], colors_all['orange'], colors_all['green_light'], colors_all['brown']]
    for res, clr, lab in zip([rms, rms_mibi, rms_maldi, rms_ngs], clrs, labels):
        ax.plot(n_cl, res, linestyle='dotted', color=clr, marker='o')
        lines.append(mlines.Line2D([], [], color=clr, marker='o', linestyle='dotted', label=lab))

    
    #ax.scatter(n_cl[-1], np.log(rms_media[0]), s=100, label='general media', marker='X', color=colors_all['orange'])
    #ax.scatter(n_cl[-1], np.log(rms_media[1]), s=100, label='selective media', marker='x', color=colors_all['green_light'])
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE ')
    #ax.set_title('RMSE depending on number of clusters', fontsize=15)
    #ax.set_yscale('log')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    ax.legend(handles=lines)
    ax.set_xlim(np.min(n_cl)-0.2, np.max(n_cl)+0.2)
    plt.savefig('model_paper/out/model_complexity/plot_rmse.png', bbox_inches='tight')
    plt.close()
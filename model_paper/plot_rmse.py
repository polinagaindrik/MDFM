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
rcParams['legend.fontsize'] = 13#15#
rcParams['figure.dpi'] = 500


def calculate_rmse(n, rn, path, add_name):
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path+df_name) for df_name in df_names]
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
    n_media = len(media)

    # Get 'real' model parameters used for data generation
    setup_real = fm.data.read_from_json('Result_temp_together_real.json', dir=path)
    param_ode_real = np.array(setup_real['param_ode'])
    T_x_real = np.array(setup_real['T_x'])
    s_x_real = np.array(setup_real['s_x']).reshape((n_media, -1))
    calibr_setup_real = {
            'model': fm.mdl.fusion_model2,
            'T_x': T_x_real,
            'output_path': path,
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
            'output_path': path,
            'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
            's_x': s_x,
            'media': media, 
        }
    x_count, obs_mibi_model, obs_maldi_model, obs_ngs_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)
    rms_1_sim = []
    for i in range(n):
        rms_1_sim.append(root_mean_squared_error(np.log(x_real2[:, i, :]), np.log(x_count[:, i, :])))
    rms0 = root_mean_squared_error(np.log(x_real2.flatten()), np.log(x_count.flatten()))
    rms_mibi0 = root_mean_squared_error(np.log(obs_mibi_model.flatten()), np.log(obs_mibi_real.flatten()))
    rms_maldi0 = root_mean_squared_error(obs_maldi_model.flatten(), obs_maldi_real.flatten())
    rms_ngs0 = root_mean_squared_error(obs_ngs_model.flatten(), obs_ngs_real.flatten())
    return data, rms0, rms_mibi0, rms_maldi0, rms_ngs0, rms_1_sim

if __name__ == "__main__":

    # RMSE for model complexity analysis
    n_cl = [4, 6, 8, 10, 12]
    n_media = 2
    relnoise = 0.1

    rms, rms_mibi, rms_maldi, rms_ngs, = [np.zeros((len(n_cl))) for _ in range(4)]
    rms_per_species = []
    L_0_real = np.array(fm.data.read_from_json('model_paper/out/Initial_values_x0_paper.json')['x0'])
    path_base = 'model_paper/out/model_complexity/'
    for i, n in enumerate(n_cl):
        path = path_base+f'{int(n)}_dim_{n_media}media_exp_{int(relnoise*100)}noise/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media)}media'
        data, rms[i], rms_mibi[i], rms_maldi[i], rms_ngs[i], rms_per_species0 = calculate_rmse(n, relnoise, path, add_name)
        rms_per_species.append(rms_per_species0)

    rms_per_species_T = []
    for j in range(np.max(n_cl)):
        rms0 = []
        for i in range(len(n_cl)):

            if len(rms_per_species[i]) > j:
                rms0.append(rms_per_species[i][j])
        rms_per_species_T.append(rms0)

    bact_all = data[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1['Others'] = (160 / 255, 160 / 255, 160 / 255)
    clrs1['Rest'] = (160 / 255, 160 / 255, 160 / 255)

    fig, ax = plt.subplots()
    labels = [r'log $x(t)$', r'log Plate Count', 'MALDI', 'NGS']
    clrs = [colors_all['blue1'], colors_all['orange'], colors_all['green_light'], colors_all['brown']]
    for res, clr, lab in zip([rms, rms_mibi, rms_maldi, rms_ngs], clrs, labels):
        ax.plot(n_cl, res, linestyle='dotted', color=clr, marker='o', label=lab)
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE ')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    coord_text = (0.07, 0.92)
    ax.text(*coord_text, '(a)', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.5, 0.67, 0.0, 0.0))
    ax.set_xlim(np.min(n_cl)-0.2, np.max(n_cl)+0.2)
    plt.savefig('model_paper/out/model_complexity/plot_rmse.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    for i, res in enumerate(rms_per_species_T):
        ax.plot(n_cl[-len(res):], res, linestyle='dotted', marker='o', color=clrs1[bact_all[i]], label=f'Species {i+1}')
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE ')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    ax.text(*coord_text, '(b)', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.72, 0.55, 0.0, 0.0), ncol=2)
    ax.set_xlim(np.min(n_cl)-0.2, np.max(n_cl)+0.2)
    plt.savefig('model_paper/out/model_complexity/plot_rmse_per_species.pdf', bbox_inches='tight')
    plt.close()

    # RMSE for noise vs. n_species analysis
    n_cl = [4, 6, 8, 10, 12]
    relnoise = [0., 0.1, 0.2, 0.3]
    n_media = 2
    path_base = 'model_paper/out/noise_vs_nspecies/'
    rms, rms_mibi, rms_maldi, rms_ngs = [np.zeros((len(n_cl), len(relnoise))) for _ in range(4)]
    for j, rn in enumerate(relnoise):
        for i, n in enumerate(n_cl):
            path = path_base+f'{int(rn*100)}noise/{int(n)}_dim_{n_media}media_exp_{int(rn*100)}noise/calibration/'
            add_name = f'_{int(n)}dim_{int(n_media)}media'
            data, rms[i, j], rms_mibi[i, j], rms_maldi[i, j], rms_ngs[i, j], rms_per_species0 = calculate_rmse(n, rn, path, add_name)

    addn = ['_x', '_pc', '_maldi', '_ngs']
    for res, add in zip([rms, rms_mibi, rms_maldi, rms_ngs], addn):
        fig, ax = plt.subplots()
        im = ax.imshow(res)
        fig.colorbar(im, orientation='vertical')
        ax.set_yticks(np.linspace(0, len(n_cl)-1, len(n_cl)))
        ax.set_yticklabels(n_cl, ha='right')
        ax.set_xticks(np.linspace(0, len(relnoise)-1, len(relnoise)))
        ax.set_xticklabels(relnoise, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Relative noise level', fontsize=15)
        ax.set_ylabel('Number of bacterial species', fontsize=15)
        plt.savefig(path_base+'rmse_noise_nspecies'+add+'.png', bbox_inches='tight')
        plt.close(fig)

    # RMSE for different media
    media = ['gen1', 'sel1', 'sel2', 'gen1+sel1']
    addn = ['_gen', '_sel', '_sel2', '']
    n = 6
    rn = 0.1
    n_media = [1, 1, 1, 2]
    path_base = 'model_paper/out/media_influence/'
    rms, rms_mibi, rms_maldi, rms_ngs = [np.zeros((len(media))) for _ in range(4)]
    for i, med in enumerate(media):
        path = path_base+f'{med}_media/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media[i])}media'+addn[i]
        data, rms[i], rms_mibi[i], rms_maldi[i], rms_ngs[i], rms_per_species0 = calculate_rmse(n, rn, path, add_name)
    
    addn = ['_x', '_pc', '_maldi', '_ngs']
    for res, add in zip([rms, rms_mibi, rms_maldi, rms_ngs], addn):
        fig, ax = plt.subplots()
        ax.stem(media, res)
        ax.set_xticks(np.linspace(0, len(media)-1, len(media)))
        ax.set_ylabel('RMSE', fontsize=15)
        ax.set_xlabel('Media', fontsize=15)
        ax.set_xticklabels(media, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(path_base+'rmse_media'+add+'.png', bbox_inches='tight')
        plt.close(fig)
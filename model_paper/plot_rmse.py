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
rcParams['figure.figsize'] = (7.2, 4.5)
rcParams['legend.framealpha'] = 0. 
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
    cost = df_optim2['cost'].values[-1]
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
    mibi_max =  np.nanmax(obs_mibi_real, axis=(0, 1)) #np.nanmax(obs_mibi_model)
    x_max = np.nanmax(x_real2, axis=(0, 1))#np.nanmax(x_count)
    rms_1_sim = []
    for i in range(n):
        rms_1_sim.append(root_mean_squared_error(x_real2[:, i, :],x_count[:, i, :]))
    #rms0 = root_mean_squared_error(np.log(x_real2.flatten()), np.log(x_count.flatten()))
    rms0 = root_mean_squared_error((x_real2/x_count).flatten(), (x_count/x_count).flatten())
    rms_mibi0 = root_mean_squared_error((obs_mibi_model/obs_mibi_model).flatten(), (obs_mibi_real/obs_mibi_model).flatten())
    rms_maldi0 = root_mean_squared_error(obs_maldi_model.flatten(), obs_maldi_real.flatten())#*n
    rms_ngs0 = root_mean_squared_error(obs_ngs_model.flatten(), obs_ngs_real.flatten())#*n
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    rms_param = [
        root_mean_squared_error(param_ode[:n*len(exps)].flatten(), param_ode_real[:n*len(exps)].flatten()),
        root_mean_squared_error(s_x.flatten(), s_x_real.flatten()),
        root_mean_squared_error(param_ode[n*len(exps):].flatten(), param_ode_real[n*len(exps):].flatten())
    ]
    return data, rms0, rms_mibi0, rms_maldi0, rms_ngs0, rms_1_sim, cost, np.array(rms_param)

if __name__ == "__main__":

    # RMSE for model complexity analysis
    n_cl = [4, 6, 8, 10, 12]
    n_media = 2
    relnoise = 0.1

    rms, rms_mibi, rms_maldi, rms_ngs, cost = [np.zeros((len(n_cl))) for _ in range(5)]
    rms_param = np.zeros((3, len(n_cl))) # 3: param_ode, x0, s_x
    rms_per_species = []
    L_0_real = np.array(fm.data.read_from_json('model_paper/out/Initial_values_x0_paper.json')['x0'])
    path_base = 'model_paper/out/model_complexity/'
    for i, n in enumerate(n_cl):
        path = path_base+f'{int(n)}_dim_{n_media}media_exp_{int(relnoise*100)}noise/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media)}media'
        data, rms[i], rms_mibi[i], rms_maldi[i], rms_ngs[i], rms_per_species0, cost[i], rms_param[:, i] = calculate_rmse(n, relnoise, path, add_name)
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
    #labels = [r'log $x(t)$', 'log Plate Count', 'MALDI', 'NGS']
    labels = [r'$x(t)$', 'Plate Count', 'MALDI', 'NGS']
    clrs = [colors_all['blue1'], colors_all['orange'], colors_all['green_light'], colors_all['brown']]
    for res, clr, lab in zip([rms, rms_mibi, rms_maldi, rms_ngs], clrs, labels):
        ax.plot(n_cl, res, linestyle='dotted', color=clr, marker='o', label=lab)
    #ax.plot(n_cl, 0.1*cost, linestyle='dotted', color= colors_all['purple'], marker='o', label=r'Cost $0.1J$')
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE ')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    coord_text = (0.07, 0.92)
    #ax.text(*coord_text, '(a)', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.6, 0.9), ncol=2)
    ax.set_xlim(np.min(n_cl)-0.2, np.max(n_cl)+0.2)
    ax.set_yscale('log')
    #ax.set_ylim(-0.02, 2.)
    plt.savefig('model_paper/out/model_complexity/plot_rmse.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    labels = [r'$\mathbf{p}$', r'$S$', r'$\mathbf{x}_0$']
    clrs = [colors_all['blue_bright'], colors_all['pink'], colors_all['green_dark']]
    for res, clr, lab in zip(rms_param, clrs, labels):
        ax.plot(n_cl, res, linestyle='dotted', color=clr, marker='o', label=lab)
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE ')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    coord_text = (0.07, 0.92)
    #ax.text(*coord_text, '(a)', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.45, 0.98), ncol=2)
    plt.savefig('model_paper/out/model_complexity/plot_rmse_param.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    for i, res in enumerate(rms_per_species_T):
        ax.plot(n_cl[-len(res):], res, linestyle='dotted', marker='o', color=clrs1[bact_all[i]], label=f'Species {i+1}')
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of bacterial species', r'RMSE$(x)$')
    ticks_val = n_cl
    tick_label = [f'{round(n)}' for n in n_cl]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    ax.text(*coord_text, '(b)', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.43, 0.52), ncol=2)
    ax.set_xlim(np.min(n_cl)-0.2, np.max(n_cl)+0.2)
    plt.savefig('model_paper/out/model_complexity/plot_rmse_per_species.pdf', bbox_inches='tight')
    plt.close()

    # RMSE for noise vs. n_species analysis
    n_cl = [4, 6, 8, 10, 12]
    relnoise = [0., 0.1, 0.2, 0.3]
    n_media = 2
    path_base = 'model_paper/out/noise_vs_nspecies/'
    rms, rms_mibi, rms_maldi, rms_ngs, cost = [np.zeros((len(n_cl), len(relnoise))) for _ in range(5)]
    for j, rn in enumerate(relnoise):
        for i, n in enumerate(n_cl):
            path = path_base+f'{int(rn*100)}noise/{int(n)}_dim_{n_media}media_exp_{int(rn*100)}noise/calibration/'
            add_name = f'_{int(n)}dim_{int(n_media)}media'
            data, rms[i, j], rms_mibi[i, j], rms_maldi[i, j], rms_ngs[i, j], rms_per_species0, cost[i, j], _ = calculate_rmse(n, rn, path, add_name)

    addn = ['_x', '_pc', '_maldi', '_ngs', '_cost']
    for res, add in zip([rms, rms_mibi, rms_maldi, rms_ngs, cost], addn):
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
        plt.savefig(path_base+'rmse_noise_nspecies'+add+'.pdf', bbox_inches='tight')
        plt.close(fig)

    # RMSE for different media
    media = ['gen1','gen2', 'gen3', 'sel1', 'sel2', 'sel3', 'gen1+sel1', 'gen2+sel2', 'gen3+sel3', 'sel1+sel2', 'sel1+sel3', 'sel2+sel3']
    addn = ['_gen', '_gen2', '_gen3', '_sel', '_sel2', '_sel3', '', '_gen2sel2', '_gen3sel3', '_sel1sel2', '_sel1sel3', '_sel2sel3']
    n = 6
    rn = 0.1
    n_media = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    path_base = 'model_paper/out/media_influence/'
    rms, rms_mibi, rms_maldi, rms_ngs, cost = [np.zeros((len(media))) for _ in range(5)]
    for i, med in enumerate(media):
        path = path_base+f'{med}_media/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media[i])}media'+addn[i]
        data, rms[i], rms_mibi[i], rms_maldi[i], rms_ngs[i], rms_per_species0, cost[i], _ = calculate_rmse(n, rn, path, add_name)
    
    addn = ['_x', '_pc', '_maldi', '_ngs', '_cost']
    media_red = ['general', 'selective', 'gen+sel', '2 selective']
    labels = [r'$x(t)$', 'Plate Count', 'MALDI', 'NGS', r'Cost $J$']
    clrs = [colors_all['blue1'], colors_all['orange'], colors_all['green_light'], colors_all['brown'], colors_all['pink']]
    fig1, ax1 = plt.subplots()
    for res, add, clr, lab in zip([rms, rms_mibi, rms_maldi, rms_ngs, cost], addn, clrs, labels):
        fig, ax = plt.subplots()
        ax.stem(media, res)
        ax.set_xticks(np.linspace(0, len(media)-1, len(media)))
        ax.set_ylabel('RMSE', fontsize=15)
        ax.set_xlabel('Media', fontsize=15)
        ax.set_xticklabels(media, ha='right', rotation=45)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(path_base+'rmse_media'+add+'.png', bbox_inches='tight')
        plt.close(fig)

        rms_analyzed = []
        for i in range (4):
            rms_analyzed.append([np.mean(res[i:i+3]), np.std(res[i:i+3])])
        #rms_analyzed.append([res[-1], 0])
        rms_analyzed = np.array(rms_analyzed)
        
        ax1.errorbar(media_red, rms_analyzed.T[0], yerr=rms_analyzed.T[1], fmt='o', linestyle='dotted', label=lab, color=clr)
    ax1.set_xticks(np.linspace(0, len(media_red)-1, len(media_red)))
    ax1.set_ylabel('RMSE', fontsize=15)
    ax1.set_xlabel('Media', fontsize=15)
    ax1.set_xticklabels(media_red, ha='right', rotation=30)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(bbox_to_anchor=(0.6, 0.85), ncol=2)
    plt.savefig(path_base+'rmse_media_mean.png', bbox_inches='tight')
    plt.close(fig)
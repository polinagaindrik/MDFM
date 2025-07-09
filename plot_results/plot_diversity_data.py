#!/usr/bin/env python3

import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 400
    exps_plot = [f'V{i:02d}' for i in range(1, 10)] + ['V11', 'V16', 'V18', 'V19']
    exp_temps = fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    
    exps_ngs = exps_plot#[f'V{i:02d}' for i in range(1, 13)] + [f'V{i:02d}' for i in range(15, 23)]
    cutoff0 = 0.1
    cutoff_prop = 0.0
    df_ngs = fm.data.get_all_experiments_dataframe(fm.data.read_ngs, exps_ngs, 'experiments/NGS/')
    df_ngs = fm.data.drop_column(df_ngs, ['M2'])
    df_ngs, bact_ngs = fm.data.preprocess_dataframe(df_ngs, cutoff=cutoff0, cutoff_prop=cutoff_prop, calc_prop=False)

    exps_maldi = exps_plot#[f'V{i:02d}' for i in range(1, 20)] + ['V21', 'V22'] + [f'V{i:02d}' for i in range(24, 28)]
    df_maldi = fm.data.get_all_experiments_dataframe(fm.data.read_maldi, exps_maldi, 'experiments/MALDI/')
    df_maldi = fm.data.drop_column(df_maldi, ['M2', 'VRBD'])
    df_maldi, bact_maldi = fm.data.preprocess_dataframe(df_maldi, cutoff=cutoff0, cutoff_prop=cutoff_prop, calc_prop=False)

    bact_all = fm.data.intersection(bact_ngs.values, bact_maldi.values)
    df_ngs = df_ngs.T[bact_all].T
    df_maldi = df_maldi.T[bact_all].T

    others = 'Rest'
    xlabel = 'Tag' #'day'

    df = df_maldi.T
    df[others] = 1-df_maldi.sum()
    df_maldi = df.T
    df = df_ngs.T
    df[others] = 1-df_ngs.sum()
    df_ngs = df.T
    bact_all += [others]

   # bact_all = sorted(set(list(bact_ngs.values) + list(bact_maldi.values)))
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    clrs1[others] = (160 / 255, 160 / 255, 160 / 255) # gray2 
    '''
    clrs1['Stenotrophomonas'] = blue_colors[-4]
    #clrs1['Hafnia'] = blue_colors[5]
    clrs1['Serratia'] = '#2F7B64'
    clrs1['Kurthia'] = colors_ngs[-1]
    #
    clrs1['Photobacterium'] = green_colors[2]

    clrs1['Lactobacillus'] = red_colors[1]
    clrs1['Lactococcus'] = blue_colors[3]
    clrs1['Listeria'] = '#B87333'
    clrs1['Leuconostoc'] = fm.plotting.green_colors[2]
    '''
    clrs1['Streptococcus'] = fm.plotting.green_colors[1]

    #json_dump(clrs1, 'colors_diversity.json', dir='out/')

    df_maldi_constT = df_ngs.filter(regex='V.._')
    print(sorted(list(set([s.split('_')[0] for s in df_maldi_constT.columns]))))

    # Plot MALDI:
    exps = sorted(list(set([s.split('_')[0] for s in df_maldi.columns])))
    for media in ['MRS', 'PC']:
        for exp in exps:
            dfs = df_maldi.filter(like=exp+'_').filter(like=media)
            #dfs, bact_maldi2 = preprocess_dataframe(dfs, cutoff=cutoff0, calc_prop=False)
            if len(dfs.columns) != 0:
                temp = int(dfs.columns[0].split("_")[2][:-1])
                days, obs_m, species = fm.data.get_values_from_dataframe(dfs, cutoff=0.)
                fig, ax = plt.subplots(1, 1, figsize=(5.5,6))#, sharey=True)
                fig.subplots_adjust(hspace=0.25)
                ax = fm.plotting.make_barplot(ax, days, np.array(obs_m), species, clrs1)
                ticks_val = days
                tick_label = [f'{round(day)}' for day in days]
                ax.set_xticks(ticks_val)
                ax.set_xticklabels(tick_label)
                ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
                ax.set_ylim(0., 1.01)
                ax.set_title(f'{exp} ({int(exp_temps[exp[:3]])}°C, {media})', fontsize=13)
                fig, ax = fm.plotting.set_labels(fig, ax, xlabel, 'MALDI-ToF')
                plt.savefig(f'out/plot_maldi/maldi_{media}_{exp}_{temp}Grad.png', bbox_inches='tight')
                plt.close(fig)

    # Plot ngs:
    exps = sorted(list(set([s.split('_')[0] for s in df_ngs.columns])))
    for exp in exps:
        dfs = df_ngs.filter(like=exp+'_')
        #dfs, bact_ngs2 = preprocess_dataframe(dfs, cutoff=cutoff0, calc_prop=False)
        temp = int(dfs.columns[0].split("_")[2][:-1])
        days, obs_m, species = fm.data.get_values_from_dataframe(dfs, cutoff=0.)
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 6))#, sharey=True)
        fig.subplots_adjust(hspace=0.25)
        ax = fm.plotting.make_barplot(ax, days, np.array(obs_m), species, clrs1)
        ticks_val = days
        tick_label = [f'{round(day)}' for day in days]
        ax.set_xticks(ticks_val)
        ax.set_xticklabels(tick_label)
        ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
        ax.set_ylim(0., 1.01)
        ax.set_title(f'{exp} ({int(exp_temps[exp[:3]])}°C)', fontsize=13)
        fig, ax = fm.plotting.set_labels(fig, ax, xlabel, 'NGS')
        plt.savefig(f'out/plot_ngs/ngs_{exp}_{temp}Grad.png', bbox_inches='tight')
        plt.close(fig)
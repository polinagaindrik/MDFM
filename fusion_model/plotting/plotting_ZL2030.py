#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from . import plotting_templates as plt_templ
from ..tools.dataframe_functions import merge_dfs, get_values_from_dataframe_biochem, filter_dataframe


scatter_marker = ['^', 'o', 'D', "s", "X", ".", "*", "P", "d", "x", '^', 'o', 'D', "s",]
mrkrs = {'CCD01': '^', 'CCD02': 'o', '': "x", 'RP':'D'}
lsts = {'CCD01': 'solid', 'CCD02': 'solid', '': "dashed"}


# Plotting of model prediction and/or measurement data:
def plot_all_ZL2030(temp_plot, **kwargs):
    plt_templ.plot_all(temp_plot, ('Tag', 'log Keimzahl'), templ_meas=template_plot_measurements_ZL2030, **kwargs)


def template_plot_measurements_ZL2030(ax0, temp, data, c='b', lab=''):
    scatter_marker = ['^', 'o', 'D', "s", "X", ".", "1", "2", "3", "4", "*", "P", "d", "+", "x"]
    n_plot = 0
    for d in data: 
        if d['const'][0] == temp:
            for meas, std in zip(d['obs_mean'], d['obs_std']):
                lab = d['experiment_name'].split("_")[0] + ' ' + d['experiment_name'].split("_")[1] + ' ' + d['experiment_name'].split("_")[2]
                ax0.errorbar(d['times'], np.log10(meas), fmt=scatter_marker[n_plot],  yerr=np.abs(0.43*std/meas),
                            markersize=7, color=c, label="{}".format(lab))
            n_plot += 1
    #ax0.set_title(f'Temperatur {round(temp)} Grad', fontsize=13)

def plot_biochem_measurement(df, meas, exp_temps):
    if meas != 'TBARS' or meas != 'POZ':
        exps = [f'V{i:02d}' for i in range(1, 15)] + ['V16', 'V17']
    if meas == 'TBARS' or meas == 'POZ':
        exps = [f'V{i:02d}' for i in range(1, 12)]
    mrkrs = ['^', 'o', 'D', "s", "X", ".", "x", "*", "P", "d", "+", "1", "2", "3", "4"]
    fig, ax = plt.subplots(figsize=(7., 6.))
    n_2, n_10, n_14 = 0, 0, 0
    for exp in exps:
        df0 = df.filter(like=exp)
        if exp_temps[exp] == 2.:
            col = plt_templ.colors[0]
            mrk = mrkrs[n_2]
            n_2 += 1
        elif exp_temps[exp] == 10.:
            col = plt_templ.colors[1]
            mrk = mrkrs[n_10]
            n_10 += 1
        elif exp_temps[exp]== 14.:
            col = plt_templ.colors[2]
            mrk = mrkrs[n_14]
            n_14 += 1

        df_wo = df0.filter(like=exp+'_')
        if float(exp[1:]) <= 9.:
            df00 = [df_wo]
        elif exp == 'V10':
            df_ccd01 = df0.filter(like='CCD01')
            df_ccd02 = df0.filter(like='CCD02')
            df01 = merge_dfs([df_wo, df_ccd01])
            df02 = merge_dfs([df_wo, df_ccd02])
            df00 = [df01, df02]
        elif exp == 'V11' or exp == 'V12' or exp == 'V17':
            df_ccd0 = df0.filter(like='CCD0')
            #df_gas = df_gas[df_gas.columns.drop(list(df_gas.filter(regex='M2')))]
            df01 = merge_dfs([df_wo.filter(like='_00'), df_wo.filter(like='_01'), df_ccd0])
            df00 = [df_wo, df01]
        elif exp == 'V15':
            df_slh1 = df0.filter(like='SLH01')
            df_slh2 = df0.filter(like='SLH02')
            df00 = [df_slh1, df_slh2]
        else:
            df_w = df0.filter(like=exp+'-')
            df00 = [df_wo, df_w]
            
        for fr in df00:
            if fr.columns[-1].split('_')[0][4:] == 'CCD01':
                lnst = 'dashed'
            elif fr.columns[-1].split('_')[0][4:] == 'CCD02':
                lnst = '-.'
            elif fr.columns[-1].split('_')[0][4:] == 'STOR01':
                lnst = 'dotted'
            elif fr.columns[-1].split('_')[0][4:] == 'SLH01':
                lnst = 'solid'
            elif fr.columns[-1].split('_')[0][4:] == 'SLH02':
                lnst = 'solid'
            else:
                lnst = 'solid'
        
            days, observable = get_values_from_dataframe_biochem(fr, meas)
            ax.plot(days, observable, label=fr.columns[-1].split('_')[0], marker=mrk, color=col, linestyle=lnst)
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', meas)
    ax.legend(fontsize=11., bbox_to_anchor=(1., 1.), framealpha=0.)
    ax.set_xlim(-0.5, 17.5)
    plt.savefig(f'out/plot_biochem/{meas}_M1.png', bbox_inches='tight')
    plt.close(fig)


def plot_measurements_ZL2030_Tunterbrech(df, exp_plots, mtimes=[], mestim=[], dir='', add_name=''):
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df.columns])))[0]
    fig, ax = plt.subplots(figsize=(6, 6.))
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    fig, ax = plt_templ.set_labels(fig, ax, *labels)
    for j, exp in enumerate(exp_plots):
        plt_templ.plot_measurement(ax, df.filter(like=exp+'_'), exp, plt_templ.exp_clrs[exp[:3]], ' ', plt_templ.mrkrs[exp[4:]])
        if len(mtimes) != 0 and len(mestim) != 0:
            plt_templ.plot_model_1temp(ax, mtimes, [mestim[j]], [exp], lsts[exp[4:]])
    ax.set_xlim(-0.5, 17.5)
    #ax.set_ylim(3.65, 8.15) # PC
    #ax.set_ylim(2.3, 8.6) # MRS
    #ticks_val = [int(2*j) for j in range (int(16*0.5)+1)]
    #tick_label = [f'{round(day)}' for day in ticks_val]
    #ax.set_xticklabels(tick_label)
    #ax.set_xticks(ticks_val)
    ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
    ax.set_title(media + ' Medien', fontsize=13)
    ax.legend(fontsize=12, framealpha=0.1, loc='upper left', ncol=1)
    plt.savefig(dir+add_name, bbox_inches='tight')
    plt.close(fig)


def plot_measurements_ZL2030_consttemp(ax0, temp, df, c=['b'], lab='',  media=''):
    scatter_marker = ['^', 'o', 'D', "s", "X", ".", "*", "P", "d", "x", '^', 'o', 'D', "s",]
    (df0, ) = filter_dataframe(f'{int(temp):02d}C', [df])
    exps = sorted(list(set([s.split('_')[0] for s in df0.columns])))
    #lst = ['solid' for _ in range (5)] + ['dashed' for _ in range (5)] + ['dotted' for _ in range (5)]
    lst = ['' for _ in range (len(exps))]
    for i, exp in enumerate(exps):
        if exp != 'V10':
            plt_templ.plot_measurement(ax0, df0.filter(like=exp+'_'), exp, plt_templ.exp_clrs[exp], lst[i], scatter_marker[i])
    #ax0.set_title(f'Temperatur {round(temp)} Grad', fontsize=13)
    if media == 'PC':
        y0_loc = 3.5
    elif media == 'MRS':
        y0_loc = 2.4
    if temp == 2:
        ax0.text(13.6, y0_loc, f'a) {round(temp)}°C', fontsize=20)
    elif temp ==10:
        ax0.text(7.7, y0_loc, f'b) {round(temp)}°C', fontsize=20)
    else:
        ax0.text(7.7, y0_loc, f'c) {round(temp)}°C', fontsize=20)

def plot_measurements_industry(df, exp_plots, dir='', add_name=''):
    clrs_ind = {'V15-SLH01': plt_templ.blue_colors[10],
                'V15-SLH02': plt_templ.blue_colors[11],
                'V21-SLH03a':plt_templ.blue_colors[1],
                'V21-SLH03b':plt_templ.blue_colors[5],
                'V23-SLH04': plt_templ.blue_colors[7],
                'V24-SLH04': plt_templ.blue_colors[0],
                'V25-SLH04': plt_templ.blue_colors[2],
                'V26-SLH04': plt_templ.blue_colors[3],
                'V27-SLH04': plt_templ.blue_colors[4],
                'V28-SLH04': plt_templ.blue_colors[6]}
    fig, ax = plt.subplots(figsize=(6, 4.))
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', 'log Keimzahl')
    for exp in exp_plots:
        exp_spl = exp.split('-')
        if len(exp_spl) >= 3:
            if exp_spl[2] == 'CCD01':
                Tshift = exp_spl[2]
            if  exp_spl[-1] == 'RP':
                mrk = 'x'
        else:
            Tshift = ''
            mrk ='o'
        plt_templ.plot_measurement(ax, df.filter(like=exp+'_'), exp, clrs_ind["-".join(exp_spl[:2])], lsts[Tshift], mrk)
    ax.set_xlim(-0.5, 17.5)
    #ax.set_ylim(3.65, 8.15) # PC
    #ax.set_ylim(1.9, 8.8) # MRS
    ticks_val = [int(2*j) for j in range (int(16*0.5)+1)]
    tick_label = [f'{round(day)}' for day in ticks_val]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
    ax.legend(fontsize=10, framealpha=0.1, loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    plt.savefig(dir+add_name, bbox_inches='tight')
    plt.close(fig)


def plot_measurements_stored_meat(df, exp_plots, dir='',  add_name=''):
    fig, ax = plt.subplots(figsize=(6, 4.))
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', 'log Keimzahl')
    lst_stor = {'STOR01': 'dashed', '': 'solid'}
    for exp in exp_plots:
        plt_templ.plot_measurement(ax, df.filter(like=exp+'_'), exp, plt_templ.exp_clrs[exp[:3]], lst_stor[exp[4:]], 'o')
    ax.set_xlim(-0.5, 17.5)
    #ax.set_ylim(4.2, 10) # PC
    ax.set_ylim(2.8, 10) # MRS
    ticks_val = [int(2*j) for j in range (int(16*0.5)+1)]
    tick_label = [f'{round(day)}' for day in ticks_val]
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_label)
    ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5] 
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12, framealpha=0.1, loc='upper left', ncol=2)
    plt.savefig(dir+add_name, bbox_inches='tight')
    plt.close(fig)

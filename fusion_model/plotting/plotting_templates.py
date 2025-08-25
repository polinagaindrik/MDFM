#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

# Set some custom colors/scatter markers for plotting (mb add more or remove later)
colors = [(0 / 255, 102 / 255, 217 / 255),
          (90 / 255, 170 / 255, 90 / 255),
          (215 / 255, 48 / 255, 39 / 255)]  
gray = (128 / 255, 128 / 255, 128 / 255)
gray2 = (160 / 255, 160 / 255, 160 / 255)
scatter_marker = ['^', 'o', 'D', "s", "X", ".", "1", "2", "3", "4", "*", "P", "d", "+", "x"]

colors_ngs = np.array([[230, 25, 75],  [228, 188, 92], [245, 130, 48], [60, 180, 75],  [0, 130, 200],
                           [145, 30, 180], [70, 240, 240], [210, 81, 49], [210, 245, 60], [250, 190, 212],
                           [0, 128, 128],  [220, 190, 255],[170, 110, 40], [255, 250, 200],[128, 0, 0],
                           [170, 255, 195],[128, 128, 0],  [255, 215, 180],[0, 0, 128],  [112, 12, 65],
                           [128, 128, 128],
                           [69, 64, 64],   [235, 236, 229],[244, 143, 40], [225, 217, 27], [18, 176, 44], 
                           [212, 131, 143],[214, 171, 177],[144,163,178],[49, 39, 54],   [165, 64, 84],
                           [249, 102, 80], [129, 203, 248],[70, 119, 218], [39, 147, 232], [85, 153, 0], 
                           [107, 147, 98], [134, 171, 165], [234, 158, 133], [84, 57, 44],
                           [196, 255, 235], [255, 225, 25],
                           [230, 25, 75],  [245, 130, 48], [60, 180, 75], [0, 130, 200],
                           [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
                           [0, 128, 128],  [220, 190, 255], [112, 12, 65], [170, 110, 40], [255, 250, 200],
                           [128, 0, 0], 
                           [170, 255, 195],[128, 128, 0],  [255, 215, 180],[0, 0, 128],    [128, 128, 128],
                           [69, 64, 64],   [235, 236, 229],[244, 143, 40], [225, 217, 27], [18, 176, 44],])
colors_ngs = colors_ngs/ 255

blue_colors = [
    '#00FFFF', '#89CFF0', '#0000FF', '#7393B3', '#0096FF', '#0047AB', '#6495ED', '#00008B', '#5D3FD3', '#A7C7E7',
    '#4169E1', '#008080', '#000080'
]
green_colors = ['#B4C424', '#40826D', '#4CBB17', '#2AAA8A', '#228B22']
red_colors = ['#EC5800', '#D2042D', '#880808', ]
colors_temps = [blue_colors, green_colors, red_colors]

exp_clrs = {'V01': blue_colors[0], 'V07': blue_colors[9], 'V08': blue_colors[10],
            'V10': blue_colors[0], 'V11':blue_colors[1],  'V12':blue_colors[2],
            'V17': blue_colors[3], 'V18':blue_colors[4],  'V19':blue_colors[5], 'V20':blue_colors[11],
            'V13': blue_colors[6], 'V14':blue_colors[7],  'V16':blue_colors[8],
            'V02': green_colors[0],'V05': green_colors[1],'V09': green_colors[2],
            'V03': red_colors[0],  'V04': red_colors[1],  'V06': red_colors[2],
             }

exp_mrkrs = {'CCD01': '^', 'CCD02': "x", '': 'o', 'STOR01':'.',
             'SLH01':'o', 'SLH02':'o', 'SLH03a1':'o', 'SLH03b':'o'}
exp_lsts = {'CCD01': 'dashed', 'CCD02': 'dashed', '': "solid", 'STOR01':'dotted',
            'SLH01':"solid", 'SLH02':"solid", 'SLH03a1':"solid", 'SLH03b':"solid"}

def template_plot_measurements0(ax0, temp, data, c='b'):
    n_plot = 0
    for d in data: 
        if d['const'] == temp:
            for meas, std in zip(d['obs_mean'], d['obs_std']):
                ax0.errorbar(d['times'], np.log10(meas), fmt=scatter_marker[n_plot],  yerr=np.abs(0.43*std/meas),
                             markersize=7, color=c, label="T = {}".format(temp))
            n_plot += 1

  
# Plotting of model prediction and/or measurement data:
def plot_all_crittime(temp_plot, labels, templ_meas=template_plot_measurements0, df=[], mtimes=[], mestim=[], mlow=[], mupp=[], tcr=[], n_cutoff=None, time_lim=[], 
             dir='', add_name='', lab_model=False):
    ncols = [2, 1, 1,]
    if len(temp_plot) == 1 or len(np.shape(temp_plot)) == 2:
        fig, ax = template_fig_1_temp(temp_plot, *labels, tcr, time_lim=time_lim)
        mtimes, mestim, mlow, mupp = [mtimes], [mestim], [mlow], [mupp]
    elif len(temp_plot) > 1 and len(np.shape(temp_plot)) == 1:
        fig, ax = template_fig_for_many_temps(temp_plot, *labels, time_lim=time_lim)

    for i, (ax0, temp) in enumerate(zip(ax, temp_plot)):
        if np.min([len(mtimes), len(mestim), len(mlow), len(mupp)]) != 0:
            if lab_model == True:  # noqa: E712
                lab = 'Modell'
                labub = r'Modellunsicherheit (95\%)'
            else:
                lab=''
                labub=''
            template_plot_model(ax0, mtimes[i], mestim[i], mlow[i], mupp[i], colors_temps[i], lab=lab, labub=labub)
            if len(tcr) != 0 and n_cutoff is not None:
                template_plot_crit_time(ax0, tcr[i], np.log10(n_cutoff), np.log10(mlow[i]))
        templ_meas(ax0, temp, df, colors_temps[i])
        ax0.legend(fontsize=14, framealpha=0.1, ncol=ncols[i], loc='upper left')
    plt.savefig(dir + f'{add_name}.png', bbox_inches='tight')
    plt.close(fig) 


# Plotting of model prediction and/or measurement data:
def plot_all(temp_plot, labels, templ_meas=template_plot_measurements0, df=[], temps=[], mtimes=[], mestim=[], time_lim=[], tcr=[], dir='', add_name=''):
    ncols = [1, 1, 1,]
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df.columns])))[0]
    exps = sorted(list(set([s.split('_')[0] for s in df.columns])))
    if len(temp_plot) == 1 or len(np.shape(temp_plot)) == 2:
        fig, ax = template_fig_1_temp(temp_plot, *labels, tcr, time_lim=time_lim)
        #mtimes, mestim = [mtimes], [mestim]
    elif len(temp_plot) > 1 and len(np.shape(temp_plot)) == 1:
        fig, ax = template_fig_for_many_temps(temp_plot, *labels, time_lim=time_lim)

    for i, (ax0, temp) in enumerate(zip(ax, temp_plot)):
        if len(temp_plot) == 1:
            plot_model_1temp(ax0, mtimes, mestim, exps, 'solid')
        else:
            plot_model_manytemps(ax0, temp, mtimes, mestim, exps, temps, 'solid')
        templ_meas(ax0, temp, df, c=colors_temps[i], media=media)
        ax0.legend(fontsize=16, framealpha=0.1, ncol=ncols[i], loc='best')
    plt.savefig(dir + f'{add_name}.png', bbox_inches='tight')
    print(dir + f'{add_name}.png')
    plt.close(fig) 


def template_fig_for_many_temps(temp_plot, xlabel, ylabel, time_lim=[]):
    fig, ax = plt.subplots(1, len(temp_plot), figsize=(6.*len(temp_plot), 6), sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.12)
    for i, ax0 in enumerate(ax): 
        ax0.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
        if len(time_lim) == len(temp_plot):
            ax0.set_xlim(-0.2, time_lim[i])
        fig, ax0 = set_labels(fig, ax0, xlabel, '')
    fig, ax[0] = set_labels(fig, ax[0], xlabel, ylabel)
    return fig, ax


def template_fig_1_temp(temp_plot, xlabel, ylabel, tcr, time_lim=[]):
    fig, ax = plt.subplots()
    if len(time_lim) != 0:
        ax.set_xlim(-0.2, time_lim[0])
        ticks_val = [int(4*j) for j in range (int(time_lim[0]*0.25)+1)]
        tick_label = [f'{round(day)}' for day in ticks_val]
        ax.set_xticks(ticks_val)
        ax.set_xticklabels(tick_label)
        ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
        if len(tcr) != 0:
            ax.set_xticks(tcr, minor = True)
            ax.set_xticklabels([f"{round(t)}" for t in tcr], minor=True)
            ax.tick_params(axis='x', which='minor', labelsize=15, labelcolor=colors[-1], labelrotation=0)
    fig, ax = set_labels(fig, ax, xlabel, ylabel)
    return fig, [ax]


def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=17)
    ax.tick_params(labelsize=15)
    ax.set_ylabel(y_label, fontsize=17)
    return fig, ax


def template_plot_model(ax0, time, estim, low, upp, c='b', lab='', labub=''):
    for tr_est, tr_u, tr_b in zip(estim, upp, low):
        ax0.plot(time, np.log10(tr_est), linestyle='solid', color=c, label=lab)
        ax0.plot(time, np.log10(tr_u), linestyle='dashed', color=gray2, label=labub)
        ax0.plot(time, np.log10(tr_b), linestyle='dashed', color=gray)


def template_plot_crit_time(ax0, tcr, n_cutoff, mlow):
    npoints = 100
    ax0.plot(tcr*np.ones((npoints)), np.linspace(np.min(mlow)*0.9, n_cutoff, npoints), linestyle='dotted', color=colors[-1],
             label="Kritische Zeit")
    ax0.plot(np.linspace(-0.2, tcr, npoints), n_cutoff*np.ones((npoints)), linestyle='dotted', color=colors[-1])


def make_barplot(ax, times, obs_data, species, clrs, cutoff=0.):
    bottom = 0.
    for bact, obs in zip(species, obs_data):
        #p = ax.bar(times, obs, 1., label=f'{obs[0]:.2f} {bact}', bottom=bottom, color=clrs[bact])
        if np.max(obs) > cutoff:
            p = ax.bar(times, obs, 0.99, label=f'{bact}', bottom=bottom, color=clrs[bact])
            bottom += obs
            ax.bar_label(p, label_type='center', fmt=fmt_func, fontsize=10)
    #p = ax.bar(times, 1-bottom, 0.99, label=f'other', bottom=bottom, color=gray2)
    #ax.bar_label(p, label_type='center', fmt=fmt_func, fontsize=10)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.1, handlelength=1.25, fontsize=13)
    return ax

def fmt_func(x):
    if x >= 0.1:
        return f'{x:.2}'
    if x > 0.04 and x <= 0.1:
        return f'{x:.1}'
    elif np.abs(x - 1) <= 0.02:
        return '1'
    else:
        return ''
    
def make_barplot2(ax, times, obs_data, species):
    bottom = 0.
    obs_data = np.array([o/np.sum(o) for o in obs_data.T]).T
    for bact, obs in zip(species, obs_data):
        p = ax.bar(times, obs, 0.7, label=f'{bact}', bottom=bottom)
        bottom += obs
        ax.bar_label(p, label_type='center', fmt='%.2f')
    ax.legend(fontsize=14)
    return ax


def plot_measurement(ax0, df0, exp, c, lst, scatter_marker):
    days_meas = [float(f.split('_')[3]) for f in df0.columns]
    obs_meas = [df0[f]['Average'] for f in df0.columns]
    std_meas = [df0[f]['Standard deviation'] for f in df0.columns]
    days_meas, obs_meas, std_meas = zip(*sorted(zip(days_meas, obs_meas, std_meas)))
    lab = exp
    ax0.errorbar(days_meas, np.log10(np.array(obs_meas)), fmt=scatter_marker,
                yerr=np.abs(0.43*(0.1+0.15*np.array(obs_meas))/np.array(obs_meas)),#np.abs(0.43*np.array(std_meas)/np.array(obs_meas)),
                linestyle=lst, linewidth=1.5,
                markersize=7, color=c, label="{}".format(lab))

    
def plot_model_1temp(ax0, t_model, obs_model, exps, lst):
    for i, exp in enumerate(exps):
        exp_num = exp.split('-')[0]
        ax0.plot(t_model, np.log10(obs_model[i]), linestyle=lst, linewidth=2., markersize=7, color=exp_clrs[exp_num])#,
                #label="{} (Modell)".format(exp))
        '''
        ax0.plot(t_model, obs_model[i], linestyle='solid', linewidth=2., markersize=7, color=exp_clrs[exp],
                label="{}".format(exp))
        '''
                       
def plot_model_manytemps(ax0, temp, t_model, obs_model, exps, temps, lst):
    for i, (exp, T) in enumerate(zip(exps, temps)):
        if T == temp:
            ax0.plot(t_model, np.log10(obs_model[i]), linestyle=lst, linewidth=2., markersize=7, color=exp_clrs[exp])#,
                    #label="{}".format(exp))
            '''
            ax0.plot(t_model, obs_model[i], linestyle='solid', linewidth=2., markersize=7, color=exp_clrs[exp],
                    label="{}".format(exp))
            '''
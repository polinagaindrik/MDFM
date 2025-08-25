import matplotlib.pyplot as plt
import numpy as np

from . import plotting_templates as plt_templ
from ..parameter_estimation.media_matrix import Scost_ngsterm, Scost_s0term
import fusion_model.tools.dataframe_functions as dtf
import fusion_model.model as mdl
#from ..dimension_reduction import *


def define_bacteria_colors(df_ngs, df_maldi):
    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=0.)
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=0.)
    bact_all = sorted(set(list(bact_ngs.values) + list(bact_maldi.values)))
    clrs1 = {}
    for b, c in zip(bact_all, plt_templ.colors_ngs[0:]):
        clrs1[b] = c
    return clrs1


def plot_filteringS2(dfs, opt_param, T_x, exps, path='', clrs=None):
    (df_mibi, df_maldi, df_ngs) = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    days_total, [obs_mibi, obs_maldi, obs_ngs] = dtf.extract_observables_from_df(dfs)
    mibi_max = np.nanmax(obs_mibi, axis=(0, 1))
    for k, exp in enumerate(exps):
        rhs1, lhs1 = Scost_ngsterm(days_total, obs_maldi[k], obs_ngs[k], opt_param, T_x)
        rhs2, lhs2 = Scost_s0term(days_total, obs_mibi[k]/mibi_max, obs_maldi[k], opt_param, 1.)
        for i, med in enumerate (media):
            fig, ax = plt.subplots(figsize=(5, 5))
            for j, bact in enumerate(df_maldi.T.columns):
                ax.scatter(np.array(days_total)[~np.isnan(rhs1[i, j])], rhs1[i, j][~np.isnan(rhs1[i, j])], color=plt_templ.colors_ngs[j], s=60, label=f'{bact}') # ({med})
                ax.plot(np.array(days_total)[~np.isnan(lhs1[i, j])], lhs1[i, j][~np.isnan(lhs1[i, j])], linewidth=2., color=plt_templ.colors_ngs[j])
            #ax.set_title('Cost 1')
            fig, ax = plt_templ.set_labels(fig, ax, 'day', r'$J_1$')
            #ax.set_ylim(-0.01, 0.2)
            ax.legend(fontsize=17, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, 1)) # 
            plt.savefig(path+exp+'_'+med+'_cost1.png', bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        for j, bact in enumerate(df_ngs.T.columns):
            ax.scatter(np.array(days_total)[~np.isnan(rhs2[j])], rhs2[j][~np.isnan(rhs2[j])], color=plt_templ.colors_ngs[j], s=60, label=f'{bact}') # ({med})
            ax.plot(np.array(days_total)[~np.isnan(lhs2[j])], lhs2[j][~np.isnan(lhs2[j])], linewidth=2., color=plt_templ.colors_ngs[j])
        fig, ax = plt_templ.set_labels(fig, ax, 'day', r'$J_2$')
        ax.legend(fontsize=17, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, 1)) # 
        plt.savefig(path+exp+'_cost2.png', bbox_inches='tight')
        plt.close(fig)
    

def plot_optimization_result(param_x0, calibr_setup, t_model, **kwargs):
    data = calibr_setup['dfs']
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(data[1])[0]
    param_opt = param_x0[n_cl*len(exps):]
    for i, exp in enumerate(exps):
        plot_one_exp_model(exp, param_opt, 10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), calibr_setup, t_model, **kwargs)


def plot_prediction_result(param, x0_dict, prediction_setup, t_model, **kwargs):
    exps = prediction_setup['exps']
    param_opt = param
    L0_arr = [10**np.array(x0_dict[exp]) for exp in exps]
    for i, exp in enumerate(exps):
        plot_one_exp_model(exp, param_opt, L0_arr[i], prediction_setup, t_model, **kwargs)


def plot_prediction_result_x0lambda(param, lambda_dict, x0_dict, prediction_setup, t_model, **kwargs):
    exps = prediction_setup['exps']
    L0_arr = [10**np.array(x0_dict[exp]) for exp in exps]
    for i, exp in enumerate(exps):
        param_opt = np.concatenate((lambda_dict[exp], param))
        plot_one_exp_model(exp, param_opt, L0_arr[i], prediction_setup, t_model, **kwargs)


def plot_one_exp_model(exp, param_opt, L0, prediction_setup, t_model, clrs=None, path='', add_name=''):
    data = prediction_setup['dfs']
    n_cl = np.shape(data[1])[0]

    (df_mibi0, df_maldi0, df_ngs0, ) = dtf.filter_dataframe(exp+'_', data)
    #temp = float(df_mibi0.columns[0].split("_")[2][:-1])
    temp = prediction_setup['exp_temps'][exp]
    if temp == 10. or temp == 14.:
        t_model = np.linspace(0., 10.+0.7, 100)
    else:
        t_model = np.linspace(0., 17.+0.7, 100)
    const = [[temp], n_cl]
    C0_opt = np.concatenate((L0, np.ones((n_cl+1))))       
    C = mdl.model_ODE_solution(prediction_setup['model'], t_model, param_opt, C0_opt, const)

    n_C = mdl.get_bacterial_count(C, n_cl, 2)
    n_C0 = mdl.get_bacterial_count(np.array(C0_opt).reshape(len(C0_opt), 1), n_cl, 2)

    days_ngs, obs_ngs = mdl.observable_NGS(t_model, n_C, prediction_setup['T_x'], n_C0, const, t_model)
    days_maldi, obs_maldi = mdl.observable_MALDI(t_model, n_C, prediction_setup['s_x'], n_C0, const, t_model)
    days_mibi, obs_mibi = mdl.observable_MiBi(t_model, n_C, prediction_setup['s_x'], n_C0, const, t_model)

    f_x = mdl.media_filtering(t_model, n_C, prediction_setup['s_x'], n_C0, const)

    plot_opt_res_realx(df_ngs0, t_model, n_C, exp, path=path+'Param_estim_', clrs=clrs, add_name=add_name)
    plot_opt_res_realx(df_ngs0, t_model, f_x[0], exp, path=path+'Sx_media1_', clrs=clrs, add_name=add_name)
    plot_opt_res_realx(df_ngs0, t_model, f_x[1], exp, path=path+'Sx_media2_', clrs=clrs, add_name=add_name)
    plot_opt_res_ngs(df_ngs0, days_ngs, obs_ngs, exp, path=path+'Param_estim_', clrs=clrs, add_name=add_name)
    plot_opt_res_maldi(df_maldi0, days_maldi, obs_maldi, exp, media=['MRS', 'PC'], path=path+'Param_estim_', clrs=clrs, add_name=add_name)
    plot_opt_res_mibi(df_mibi0, days_mibi, obs_mibi, exp, media=['MRS', 'PC'], path=path+'Param_estim_', add_name=add_name)


def plot_opt_res_ngs(df_ngs0, days_model, obs_model, exp, std=None, path='', add_name='', clrs=None):
    fig, ax = plt.subplots(figsize=(5, 4.5)) # (4.5, 5)
    fig.subplots_adjust()
    days_meas = [float(f.split('_')[3]) for f in df_ngs0.columns]
    obs_meas = np.array([df_ngs0[f] for f in df_ngs0.columns])
    days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))
    for k, (o_n, o_meas, bact) in enumerate(zip(np.array(obs_model), np.array(obs_meas).T, df_ngs0.T.columns)):
        if std is None:
            std = 0.1 + o_meas*0.15
        if clrs is not None:
            clr = clrs[bact]
        else:
            clr = plt_templ.colors_ngs[4*k]
        ax.errorbar(days_meas, o_meas, yerr=std, fmt='o', color=clr, markersize=8, label=f'{bact}') # , label=f'Data Cl. {k}'
        #ax.scatter(days_model, o_n, linewidth=2., label=f'{bact}', color='w', marker='o', s=40, edgecolors=clr)
        ax.plot(days_model, o_n, linewidth=2., color=clr)      
    ax.set_title(f'NGS, {exp}', fontsize=15)
    #fig, ax = set_labels(fig, ax, 'day', r'$T \mathbf{x   } / \lVert T \mathbf{x} \rVert$')
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'$T \mathbf{x   } / \lVert T \mathbf{x} \rVert$')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(np.min(days_meas)-0.7, np.max(days_meas)+0.7)
    ax.legend(fontsize=13, framealpha=0., handlelength=1.5, bbox_to_anchor=(1.,1.))# bbox_to_anchor=(1., 0.1, 0.5, 0.5))#, bbox_to_anchor=(1, 1)) # 
    plt.savefig(path+exp+add_name+'_ngs.png', bbox_inches='tight')
    plt.close(fig)


def plot_opt_res_maldi(df_maldi0, days_model, obs_model, exp, std=None, media=[' '], path='', add_name='', clrs=None):
    #lnsts = ['solid', 'dashed', '-.']
    for j, med in enumerate(media):
        fig, ax = plt.subplots(figsize=(5, 4.5))  # (4.5, 5)
        fig.subplots_adjust()
        days_meas = [float(f.split('_')[3]) for f in df_maldi0.filter(like=med).columns]
        obs_meas = np.array([df_maldi0.filter(like=med)[f] for f in df_maldi0.filter(like=med).columns])
        days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))
        o_model = obs_model[j]     
        for k, (o_m, o_meas, bact) in enumerate(zip(np.array(o_model), np.array(obs_meas).T, df_maldi0.T.columns)):
            if clrs is not None:
                clr = clrs[bact]
            else:
                clr = plt_templ.colors_ngs[4*k]
            if std is None:
                std = 0.1 + o_meas*0.15
            #ax.plot(days_model, o_m, linewidth=2., label=f'{bact} ({med})', color=clr, linestyle=lnsts[j])
            ax.errorbar(days_meas, o_meas, yerr=std, fmt='o', color=clr, markersize=8, label=f'{bact} ({med})')
            #ax.scatter(days_model, o_m, linewidth=2., label=f'{bact} ({med})', color='w', marker='o', s=40, edgecolors=clr) #color='w'
            ax.plot(days_model, o_m, linewidth=2., color=clr)
        ax.set_title(f'MALDI-ToF, {exp}', fontsize=15)
        ax.set_ylim(-0.05, 1.05)
        #fig, ax = set_labels(fig, ax, 'day', r'$S \mathbf{x} / \lVert S \mathbf{x}\rVert$')
        fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'$S \mathbf{x} / \lVert S \mathbf{x}\rVert$')
        ax.legend(fontsize=13, framealpha=0., handlelength=1.5, bbox_to_anchor=(1., 1.))#, bbox_to_anchor=(1., 0.1, 0.5, 0.5)) #, bbox_to_anchor=(1, 1)
        ax.set_xlim(np.min(days_meas)-0.7, np.max(days_meas)+0.7)
        plt.savefig(path+exp+add_name+f'_{med}'+'_maldi.png', bbox_inches='tight')
        plt.close(fig)
    

def plot_opt_res_mibi(df_mibi0, days_model, obs_model, exp, std=None, media=[''], path='', add_name='', clrs=None):
    fig, ax = plt.subplots(figsize=(5, 4.5)) # (4.5, 5)
    fig.subplots_adjust()
    for j, med in enumerate(media):
        days_meas = [float(f.split('_')[3]) for f in df_mibi0.filter(like=med).columns]
        obs_meas = np.array([df_mibi0.filter(like=med)[f]['Average'] for f in df_mibi0.filter(like=med).columns])
        stds = np.array([df_mibi0.filter(like=med)[f]['Standard deviation'] for f in df_mibi0.filter(like=med).columns])
        days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))
        o_model = obs_model[j]
        #if std is None:
        #    std = #0.01 + np.array(obs_meas)*0.1
        ax.plot(days_model, o_model, label=f'Model (media {med})', linewidth=2.,  color=plt_templ.colors_ngs[2*j])
        ax.errorbar(days_meas, obs_meas, yerr=stds, fmt='o', color=plt_templ.colors_ngs[2*j]) # label=f'Data (media {j+1})',
    ax.set_title(exp)
    ax.set_title('MiBi', fontsize=15)
    ax.set_yscale('log')
    ax.set_xlim(np.min(days_meas)-0.5, np.max(days_meas)+1)
    #fig, ax = set_labels(fig, ax, 'day', r'log CFU mL$^{-1}$')
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'log CFU mL$^{-1}$')
    ax.legend(fontsize=13, framealpha=0., handlelength=1.5)
    plt.savefig(path+exp+add_name+'_mibi.png', bbox_inches='tight')
    plt.close(fig)


def plot_opt_res_realx(df0, days_model, obs_model, exp, path='', add_name='', clrs=None):
    fig, ax = plt.subplots(figsize=(6, 4.5,))
    fig.subplots_adjust()
    bact_all = df0.T.columns
    for k, o_n in enumerate(np.array(obs_model)):
        if clrs is not None:
            clr = clrs[bact_all[k]]
        else:
            clr = plt_templ.colors_ngs[4*k]
        ax.plot(days_model, o_n, linewidth=2., label=f'{bact_all[k]}', color=clr)
    ax.set_yscale('log')
    ax.set_title('Wahrer x(t)-Wert, '+exp, fontsize=15)
    ax.set_xlim(0., np.max(days_model))
    #fig, ax = set_labels(fig, ax, 'day', r'log CFU mL$^{-1}$')
    fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'log CFU mL$^{-1}$')
    ax.legend(fontsize=13, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, 1))
    plt.savefig(path+exp+add_name+'_realx.png', bbox_inches='tight')
    plt.close(fig)
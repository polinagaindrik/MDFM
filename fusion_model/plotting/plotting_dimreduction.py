import matplotlib.pyplot as plt
import numpy as np

from ..tools.dataframe_functions import get_values_from_dataframe, filter_dataframe
from ..data.output import read_from_json
from ..model.solving import model_ODE_solution, get_bacterial_count
from ..dimensionality_reduction.dimension_reduction import model_one_experiment_withP
from ..data.read_ZL2030data import read_pkl
from .plotting_templates import set_labels, colors_ngs
from .plotting_results import define_bacteria_colors, plot_opt_res_ngs, plot_opt_res_maldi, plot_opt_res_mibi


def lin_func(x, k, b):
    return b + k * x


def plot_p_slopes(exps, path=''):
    Pmatrix_separate = read_from_json('Pmatrix_temp_separate.json', dir=path)['P_matrix']
    Pmatrix_together = read_from_json('Pmatrix_temp_together.json', dir=path)['P_matrix']
    x_plot = np.linspace(0, 1000, 50)
    lwdth = [1.7, 1.5, 2., 3.]
    lbls = [r'$P_{2C}$', r'$P_{10C}$', r'$P_{14C}$', r'$P_{tot}$']

    df_x = read_pkl('dataframe_x.pkl', path) 
    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    for i, p_vals in enumerate(zip(*Pmatrix_separate, Pmatrix_together)):
        for j, p  in enumerate(p_vals):
            ax[i, 0].plot(lin_func(x_plot,p[0], 0), lin_func(x_plot, p[1], 0), label=lbls[j], linewidth=lwdth[j], linestyle='dashed')
            ax[i, 1].plot(lin_func(x_plot,p[1], 0), lin_func(x_plot, p[2], 0), label=lbls[j], linewidth=lwdth[j], linestyle='dashed')
            ax[i, 2].plot(lin_func(x_plot,p[2], 0), lin_func(x_plot, p[3], 0), label=lbls[j], linewidth=lwdth[j], linestyle='dashed')
            for j in range (3):
                ax[i, j].set_ylim(-0.02, 1)
                ax[i, j].set_xlim(-0.02, 1)
    for exp in exps:
        counts = df_x.filter(like=exp).values
        ax[2, 0].plot(counts[0], counts[1], label=exp, marker='o')
        ax[2, 1].plot(counts[1], counts[2], label=exp, marker='o')
        ax[2, 2].plot(counts[2], counts[3], label=exp, marker='o')
    for i in range (3):
        for j in range (3):
            ax[i, j].legend(fontsize=13, framealpha=0., handlelength=1.5)
            fig, ax[i, 0] = set_labels(fig, ax[i, 0], r'$x_0$', r'$x_1$')
            fig, ax[i, 1] = set_labels(fig, ax[i, 1], r'$x_1$', r'$x_2$')
            fig, ax[i, 2] = set_labels(fig, ax[i, 2], r'$x_2$', r'$x_3$')
    ax[0, 1].set_title('Cluster 0')
    ax[1, 1].set_title('Cluster 1')
    plt.savefig(path+'Pmatrix.png', bbox_inches='tight')
    plt.close(fig)


##### P matrix results ##########
def plot_proj_matrix_result(df_ngs, df_ngs2, df_maldi, df_maldi2, path='', add_name='', media=['PC', 'MRS']):
    exps = sorted(list(set([s.split('_')[0] for s in df_ngs.columns])))
    clrs = define_bacteria_colors(df_ngs, df_maldi)
    #clrs = read_from_json('out/colors_diversity.json')
    for exp in exps:
        df_ngs0, df_ngs_model, df_maldi0, df_maldi_model = filter_dataframe(exp, [df_ngs, df_ngs2, df_maldi, df_maldi2])
        plot_opt_res_ngs(df_ngs0, df_ngs_model, exp, path=path+'Pestim_', add_name=add_name, clrs=clrs)
        plot_opt_res_maldi(df_maldi0, df_maldi_model, exp, media=[media[0]], path=path+f'Pestim_{media[0]}_', add_name=add_name, clrs=clrs)
        plot_opt_res_maldi(df_maldi0, df_maldi_model, exp, media=[media[1]], path=path+f'Pestim_{media[1]}_', add_name=add_name, clrs=clrs)
        #plot_opt_res_maldi(df_maldi0, df_maldi_model, exp, media=media, path=path+f'Pestim_', add_name=add_name, clrs=clrs)


def plot_optimization_result_withP(data, param_x0, calibr_setup, s_x, t_model, path=''):
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    Pmatrix = calibr_setup['P_matrix']
    Pinv = np.linalg.pinv(Pmatrix)[0]
    n_cl = np.shape(Pmatrix)[0]
    param_opt = param_x0[n_cl*len(exps):]
    print('x0:', param_x0[:n_cl*len(exps)])
    print('p:', param_x0[n_cl*len(exps):])
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], n_cl]
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        df_mibi_model, df_maldi_model, df_ngs_model = model_one_experiment_withP(df_mibi0, df_maldi0, df_ngs0, calibr_setup['model'], t_model,
                                                                                 param_opt, C0_opt, const, Pinv, s_x, t_model, t_model,
                                                                                 n_states=2)
        # Calculate model + Observables calculation: MiBi, MALDI, NGS
        plot_opt_res_mibi(df_mibi0, df_mibi_model, exp, path=path+'Param_estim_', media=['PC', 'MRS'])
        plot_opt_res_maldi(df_maldi0, df_maldi_model, exp, path=path+'Param_estim_', media=['PC', 'MRS'])
        plot_opt_res_ngs(df_ngs0, df_ngs_model, exp, path=path+'Param_estim_')


def plot_optimization_result_withP_diffT(data, param_x0, calibr_setup, s_x, t_model, path=''):
    Pmatrices = calibr_setup['P_matrix']
    Pinv = [np.linalg.pinv(p) for p in Pmatrices]
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(Pmatrices[0])[0]
    param_opt = param_x0[n_cl*len(exps):]
    print('x0:', param_x0[:n_cl*len(exps)])
    print('p:', param_x0[n_cl*len(exps):])
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], np.shape(Pmatrices[0])[0]]
        if temp == 2.:
            pinv = Pinv[0]
        elif temp == 10.:
            pinv = Pinv[1]
        elif temp == 14.:
            pinv = Pinv[2]
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        df_mibi_model, df_maldi_model, df_ngs_model = model_one_experiment_withP(df_mibi0, df_maldi0, df_ngs0, calibr_setup['model'], t_model,
                                                                                 param_opt, C0_opt, const, pinv, s_x, t_model, t_model,
                                                                                 n_states=2)
        # Calculate model + Observables calculation: MiBi, MALDI, NGS
        plot_opt_res_mibi(df_mibi0, df_mibi_model, exp, path=path+'Param_estim_diffT_', media=['PC', 'MRS'])
        plot_opt_res_maldi(df_maldi0, df_maldi_model, exp, path=path+'Param_estim_diffT_', media=['PC', 'MRS'])
        plot_opt_res_ngs(df_ngs0, df_ngs_model, exp, path=path+'Param_estim_diffT_')


def plot_reduced_model(data, param_x0, calibr_setup, s_x, t_model, path=''):
    Pmatrices = calibr_setup['P_matrix']
    Pinv = [np.linalg.pinv(p) for p in Pmatrices]
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(Pmatrices[0])[0]
    param_opt = param_x0[n_cl*len(exps):]
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], np.shape(Pmatrices[0])[0]]
        if temp == 2.:
            pinv = Pinv[0]
        elif temp == 10.:
            pinv = Pinv[1]
        elif temp == 14.:
            pinv = Pinv[2]
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(calibr_setup['model'], t_model, param_opt, C0_opt, const)
        n_C = get_bacterial_count(C, np.shape(pinv)[1], 2)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.subplots_adjust()
        for k, o_n in enumerate(np.array(n_C)):
            ax.plot(t_model, o_n, linewidth=2., label=f'Model Cl. {k}', color=colors_ngs[4*k])
        ax.set_yscale('log')
        ax.set_title(exp, fontsize=15)
        ax.set_xlim(-0.2, 17.3)
        fig, ax = set_labels(fig, ax, 'day', r'log CFU mL$^{-1}$')
        ax.legend(fontsize=13, framealpha=0., handlelength=1.5)
        plt.savefig(path+f'Reduced_model_{exp}_realx.png', bbox_inches='tight')
        plt.close(fig)


def plot_reduced_model_temptogether(data, param_x0, calibr_setup, s_x, t_model, path=''):
    Pmatrix = calibr_setup['P_matrix']
    Pinv = np.linalg.pinv(Pmatrix)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(Pmatrix)[0]
    param_opt = param_x0[n_cl*len(exps):]
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], np.shape(Pmatrix)[0]]
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(calibr_setup['model'], t_model, param_opt, C0_opt, const)
        n_C = get_bacterial_count(C, np.shape(Pinv)[1], 2)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.subplots_adjust()
        for k, o_n in enumerate(np.array(n_C)):
            ax.plot(t_model, o_n, linewidth=2., label=f'Model Cl. {k}', color=colors_ngs[4*k])
        ax.set_yscale('log')
        ax.set_title(exp, fontsize=15)
        ax.set_xlim(-0.2, 17.3)
        fig, ax = set_labels(fig, ax, 'day', r'log CFU mL$^{-1}$')
        ax.legend(fontsize=13, framealpha=0., handlelength=1.5)
        plt.savefig(path+f'Reduced_model_{exp}_realx.png', bbox_inches='tight')
        plt.close(fig)


def plot_reduced_model_direct(data, param_x0, calibr_setup, s_x, t_model, path=''):
    n_cl = np.shape(s_x)[1]
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    param_opt = param_x0[n_cl*len(exps):]
    for i, exp in enumerate(exps):
        df_mibi0, df_maldi0, df_ngs0 = filter_dataframe(exp, data)
        temp = float(df_mibi0.columns[0].split("_")[2][:-1])
        const = [[temp], n_cl]
        C0_opt = np.concatenate((10**np.array(param_x0[n_cl*i:n_cl*(i+1)]), np.ones((n_cl+1))))
        C = model_ODE_solution(calibr_setup['model'], t_model, param_opt, C0_opt, const)
        n_C = get_bacterial_count(C, n_cl, 2)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.subplots_adjust()
        for k, o_n in enumerate(np.array(n_C)):
            ax.plot(t_model, o_n, linewidth=2., label=f'Model Cl. {k}', color=colors_ngs[4*k])
        ax.set_yscale('log')
        ax.set_title(exp, fontsize=15)
        ax.set_xlim(-0.2, 17.3)
        fig, ax = set_labels(fig, ax, 'day', r'log CFU mL$^{-1}$')
        ax.legend(fontsize=13, framealpha=0., handlelength=1.5)
        plt.savefig(path+f'Reduced_model_{exp}_realx.png', bbox_inches='tight')
        plt.close(fig)


def plot_clusters(df, kmeans, y):
    exps = sorted(list(set([s.split('_')[0] for s in df.columns])))
    t_counter = 0
    titles = [f'Bacteria_{i:02d}' for i in range (kmeans.n_clusters)]
    for j, exp in enumerate(exps):
        fig, ax = plt.subplots(1, kmeans.n_clusters, figsize=(10, 5))
        fig.subplots_adjust(hspace=0.3)
        days, observable, bacteria = get_values_from_dataframe(df.filter(like=exp))
        for i in range (kmeans.n_clusters):
            for yy in y:
                if yy[0] == i:
                    ax[i].plot(days, yy[-1][t_counter:t_counter+len(days)], marker='o',  label='Species_'+yy[1][0][-2:])  
            ax[i].plot(days, kmeans.cluster_centers_[i][t_counter:t_counter+len(days)], color='k', marker='o',
                       linewidth=2.1, label=f'Kmean {i}')
            ax[i].legend(fontsize=10)
            ax[i].set_title(titles[i], fontsize=13)
            if i == 0:
                fig, ax[i] = set_labels(fig, ax[i], 'day', 'NGS')
            else:
                fig, ax[i] = set_labels(fig, ax[i], 'day', '')
        t_counter += len(days)
        plt.savefig(f'out/test_insilico/kmeans_exp{exps[j]}.pdf', bbox_inches='tight')
        plt.close(fig)
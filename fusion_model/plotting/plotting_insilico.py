import matplotlib.pyplot as plt
import numpy as np
from ..model.solving import model_ODE_solution, get_bacterial_count
from . import plotting_templates as plt_templ


def plot_insilico_x(df, model, t, param, x0, n_cl, path='', add_name=''):
    t_model = np.linspace(min(t), max(t), 100)
    exps = sorted(list(set([s.split('_')[0] for s in df.columns])))
    for exp in exps:
        df0 = df.filter(like=exp)
        temp = float(df0.columns[0].split("_")[2][:-1])
        const = [[temp], n_cl, 1.] 
        x = model_ODE_solution(model, t_model, param, x0, const)
        n_x = get_bacterial_count(x, n_cl, 2)

        #plot_opt_res_realx(df0, df_model, exp, path=path, add_name=add_name)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.subplots_adjust()
        days_meas = [float(f.split('_')[3]) for f in df0.columns]
        obs_meas = np.array([df0[f]for f in df0.columns])
        days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))

        obs_model = n_x.T
        days_model = t_model
        for k, (o_n, o_meas, bact) in enumerate(zip(np.array(obs_model).T, np.array(obs_meas).T, df0.T.columns)):
            std = 0.01 + o_meas*0.1
            ax.plot(days_model, o_n, linewidth=2., label=f'{bact}', color=plt_templ.colors_ngs[4*k])
            ax.errorbar(days_meas, o_meas, yerr=std, fmt='o', color=plt_templ.colors_ngs[4*k]) #, label=f'Data Cl. {k}'
        ax.set_yscale('log')
        ax.set_title(r'In-silico model $x(t)$', fontsize=15)
        ax.set_yscale('log')
        ax.set_xlim(-0.2, 17.3)
        fig, ax = plt_templ.set_labels(fig, ax, 'day', r'$x(t)$')
        ax.legend(fontsize=13, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, .8))
        plt.savefig(path+add_name+f'{exp}_realx.pdf', bbox_inches='tight')
        plt.close(fig)
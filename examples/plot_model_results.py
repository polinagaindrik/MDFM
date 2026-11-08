import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    n_cl = 2
    n_media = 2
    relnoise = 0.
    n_exps = 3

    path = f'out/main_param_distrib2_{int(n_exps)}exp/'
    path2=path
    exp_temps = fm.output.read_from_json(''+'exp_temps_model_paper.json', dir=path2)

    # Calibration result:
    res = fm.output.read_from_json('Result_calibration.json', dir=path2)
    T_x = res['T_x']
    param_opt = res['param_ode']
    #param_ode = param_opt[:-n_cl*n_media]
    x0_vals = param_opt[:n_cl]#x0_vals = param_opt[:n_cl]*len(exp_temps)]
    lambd_opt = param_opt[n_cl:n_cl+n_cl] #lambd_opt = param_opt[n_cl*len(exp_temps):n_cl*len(exp_temps)+n_cl]
    print(f"Optimized lambda: {lambd_opt}")
    mu_opt = param_opt[n_cl + n_cl:2*n_cl + n_cl*len(exp_temps)]
    #mu_opt = param_opt[n_cl + n_cl*len(exp_temps):n_cl + 2*n_cl*len(exp_temps)]
    rest_ode_param = param_opt[2*n_cl + n_cl*len(exp_temps):]
    #rest_ode_param = param_opt[n_cl + 2*n_cl*len(exp_temps):]
    print(f"Optimized mu: {mu_opt}")
    print(f"n_max: {rest_ode_param[0]}")
    s_x = np.array(res['s_x']).reshape((n_media, n_cl))

    mu_exps = []
    mu_opt_vals = []
    for k in range (n_exps):
        add_name = f'_{k}'
        df_names = [f'dataframe_platecount{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        data = fm.dtf.filter_dataframe_regex('V.._', data)
        exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')

        res_exp = fm.output.read_from_json(f'Result_real_paramdistrib_{k}.json', dir=path2)
        param_exp = np.array(res_exp['param_ode'])
        mu_exps.append(param_exp[2*n_cl:2*n_cl+n_cl])
        lambd_exp = param_exp[n_cl:2*n_cl]
        rest_ode_param_exp = param_exp[2*n_cl+n_cl:]
        mu_opt_vals.append(mu_opt[k*n_cl:(k+1)*n_cl])


        # Plot resulting model
        calibr_setup={
                'model': fm.mdl.fusion_model_distr,
                'T_x': T_x,
                'output_path': path2,
                'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir=path2),
                's_x': s_x,
                'media': media, 
        }

        param_ode = np.concatenate((x0_vals, lambd_opt, mu_opt[k*n_cl:(k+1)*n_cl], rest_ode_param))
        t_model = np.linspace(0., 18., 100)

        x_count, obs_platecount_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(fm.dtf.filter_dataframe(f'V0{k+1}', data), param_ode, calibr_setup, t_model)
        exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
        
        '''
        exp_clrs = {}
        i, j, k = 0, 0, 0
        for exp, temp in zip(calibr_setup['exp_temps'], calibr_setup['exp_temps'].values()):
            if temp == 2.:
                exp_clrs[exp] =  fm.plotting.blue_colors[i]
                i += 1
            elif temp == 6.:
                exp_clrs[exp] =  fm.plotting.green_colors[j]
                j += 1
            else:
                exp_clrs[exp] =  fm.plotting.red_colors[k]
                k += 1
        '''
        exp_clrs = {}
        for exp in calibr_setup['exp_temps']:
            exp_clrs[exp] = fm.plotting.blue_colors[0]

        labels = ('Time',  r'log CFU mL$^{-1}$')
        '''
        for j, med in enumerate(media):
            fm.plotting.plot_all([2., 6., 10.], labels, templ_meas=fm.plotting.plot_measurements_insilico, df=data[0].filter(like=med), clrs=exp_clrs,
                    temps=temps_model, mtimes=t_model, mestim=obs_platecount_model[:,j, : ], dir=path2, add_name=f'platecount_{med}_const_model'+add_name, time_lim=[17.5, 17.5, 17.5])
        '''

        for j, med in enumerate(media):
            for i, temp in enumerate([2.,]):
                obs_platecount = obs_platecount_model[3*i:3*i+3,j,:]
                fm.plotting.plot_all([temp], labels, templ_meas=fm.plotting.plot_measurements_insilico, df=data[0].filter(like=med).filter(like=f'_{int(temp):02d}C_'), clrs=exp_clrs,temps=temps_model, mtimes=t_model, mestim=obs_platecount, dir=path2, add_name=f'platecount_{med}_const_model_{int(temp)}Grad'+add_name, time_lim=[17.5])

    # Plot mu distribution:
    fig, ax = plt.subplots()
    ax.hist(mu_opt, bins=20)  # arguments are passed to np.histogram
    plt.savefig(path2+'mu_opt_histogram.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.array(mu_exps).flatten(), bins=20)  # arguments are passed to np.histogram
    plt.savefig(path2+'mu_exps_histogram.png', bbox_inches='tight')
    plt.close(fig)

    # Calculate mean and sigma of the mu distribution
    shape, loc, scale = stats.lognorm.fit(mu_opt, floc=0)
    mean_opt, sigma_opt = np.log(scale), shape
    print(f"Optimized: Mean: {mean_opt}, Sigma: {sigma_opt}")

    # Calculate mean and sigma of the mu distribution
    shape, loc, scale = stats.lognorm.fit(np.array(mu_exps).flatten(), floc=0)
    mean, sigma = np.log(scale), shape
    print(f"Data: Mean: {mean}, Sigma: {sigma}")


    fig, ax = plt.subplots()
    x = np.linspace(0., 5, 100)
    pdf = stats.lognorm.pdf(x, s=sigma_opt, scale=np.exp(mean_opt))
    ax.plot(x, pdf, label=f'Estimated distribution\n (mean={mean_opt:.2f}, sigma={sigma_opt:.2f})')
    mean, sigma = 0.5, 0.5
    x = np.linspace(0., 5, 100)
    pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mean))
    ax.plot(x, pdf, label=f'"Real" distribution\n (mean={mean:.2f}, sigma={sigma:.2f})')
    plt.xlabel(r'Growth rate $\mu$')
    plt.legend()
    plt.savefig(path2+'mu_distribution.png', bbox_inches='tight')
    plt.close(fig)


    fig, ax = plt.subplots()
    sct_mrk = ['o', 'x', '^', 's', 'D', 'P', '*', 'h', 'v', '<', '>']
    tick_lbls = [r'$\lambda_1$', r'$\lambda_2$', r'$n_{\max}$', r'k$_{12}$', r'k$_{21}$']
    param_plot_opt = np.concatenate((lambd_opt, rest_ode_param))
    param_plot_exps = np.concatenate((lambd_exp, rest_ode_param_exp))
    ax.scatter(np.linspace(1, len(param_plot_opt), len(param_plot_opt)), param_plot_opt, marker=sct_mrk[i])
    ax.scatter(np.linspace(1, len(param_plot_exps), len(param_plot_exps)), param_plot_exps, marker=sct_mrk[i])
    ax.set_yscale('log')
    plt.legend(('Optimized', 'Exps'))
    plt.ylabel('Parameter value')
    ticks_val = np.linspace(1, len(param_plot_opt), len(param_plot_opt))
    ax.set_xticks(ticks_val)
    ax.set_xticklabels(tick_lbls)
    plt.savefig(path2+'param_opt_vs_exps.png', bbox_inches='tight')
    plt.close(fig)

    mu_exps = np.array(mu_exps)
    mu_opt_vals = np.array(mu_opt_vals)
    fig, ax = plt.subplots()
    for i in range (n_cl):
        ax.scatter(np.linspace(1, n_exps, n_exps), mu_opt_vals[:, i], marker=sct_mrk[i], label=f'Optimized {i+1} Sp.')
        ax.scatter(np.linspace(1, n_exps, n_exps), mu_exps[:, i], marker=sct_mrk[i], label=f'Exps {i+1} Sp.')
    ax.set_yscale('log')
    plt.ylabel(r'Growth rate $\mu$')
    plt.xlabel('Experiment Number')
    ticks_val = np.linspace(1, n_exps, n_exps)
    ax.set_xticks(ticks_val)
    plt.legend(framealpha=0.5, bbox_to_anchor=(1, 1.05))
    plt.savefig(path2+'mu_comparison_opt_vs_exps.png', bbox_inches='tight')
    plt.close(fig)
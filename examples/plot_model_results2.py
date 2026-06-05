import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_observables_from_df(dfs):
    (df_mibi, df_maldi, df_ngs, df_x ) = dfs
    exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))
    days_x = sorted(set([float(f.split('_')[3]) for f in df_x.columns]))
    x = fm.dtf.extract_observables_from_df_x(df_x, days_x, exps)
    return days_x , [x]


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    n_cl = 2
    n_media = 2
    relnoise = 0.

    path = 'out/main_param_distrib2/'
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
    alph_opt = param_opt[n_cl + n_cl:2*n_cl + n_cl*len(exp_temps)]
    #alph_opt = param_opt[n_cl + n_cl*len(exp_temps):n_cl + 2*n_cl*len(exp_temps)]
    rest_ode_param = param_opt[2*n_cl + n_cl*len(exp_temps):]
    #rest_ode_param = param_opt[n_cl + 2*n_cl*len(exp_temps):]
    print(f"Optimized alpha: {alph_opt}")
    print(f"n_max: {rest_ode_param[0]}")
    s_x = np.array(res['s_x']).reshape((n_media, n_cl))

    n_exps = 3
    alph_exps = []
    for k in range (n_exps):
        add_name = f'_{k}'
        df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl', f'dataframe_x{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        data = fm.dtf.filter_dataframe_regex('V.._', data)
        exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        res_exp = fm.output.read_from_json(f'Result_real_paramdistrib_{k}.json', dir=path2)
        param_exp = np.array(res_exp['param_ode'])
        alph_exps.append(param_exp[2*n_cl:2*n_cl+n_cl])
        lambd_exp = param_exp[n_cl:2*n_cl]

        '''
        step = 1
        optim_file2 = f"optimization_history{int(step)}.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        T_x = [0.] +[1. for _ in range(n_cl-1)]
        # Take optimal parameter values on last optimization step
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
        s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
        param_ode = param_opt[:-n_cl*n_media]
        '''
        # Plot resulting model
        calibr_setup={
                'model': fm.mdl.fusion_model_distr,
                'T_x': T_x,
                'output_path': path2,
                'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir=path2),
                's_x': s_x,
                'media': media, 
        }

        param_ode = np.concatenate((x0_vals, lambd_opt, alph_opt[k*n_cl:(k+1)*n_cl], rest_ode_param))
        t_model = np.linspace(0., 18., 100)

        x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(fm.dtf.filter_dataframe(f'V0{k+1}', data[:-1]), param_ode, calibr_setup, t_model)
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
                    temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{med}_const_model'+add_name, time_lim=[17.5, 17.5, 17.5])


        for j, med in enumerate(media):
            for i, temp in enumerate([2.,]):
                obs_mibi = obs_mibi_model[3*i:3*i+3,j,:]
                fm.plotting.plot_all([temp], labels, templ_meas=fm.plotting.plot_measurements_insilico, df=data[0].filter(like=med).filter(like=f'_{int(temp):02d}C_'), clrs=exp_clrs,temps=temps_model, mtimes=t_model, mestim=obs_mibi, dir=path2, add_name=f'MiBi_{med}_const_model_{int(temp)}Grad'+add_name, time_lim=[17.5])
        '''
        t_exp, [x_exp] = extract_observables_from_df(data)
        fig, ax = plt.subplots()
        for i in range (n_cl):
            ax.plot(t_model, x_count[0, i], label=f'Model cl{i+1}')
            ax.scatter(t_exp, x_exp[0, i], label=f'Exp cl{i+1}')
        ax.set_yscale('log')
        plt.savefig(path2+f'x_count_{k}exp.png', bbox_inches='tight')
        plt.close(fig) 
    # Plot alphas distribution:
    fig, ax = plt.subplots()
    ax.hist(alph_opt, bins=20)  # arguments are passed to np.histogram
    plt.savefig(path2+'alph_opt_histogram.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.array(alph_exps).flatten(), bins=20)  # arguments are passed to np.histogram
    plt.savefig(path2+'alph_exps_histogram.png', bbox_inches='tight')
    plt.close(fig)

    # Calculate mu sigma for alpha distributions
    shape, loc, scale = stats.lognorm.fit(alph_opt, floc=0)
    mu, sigma = np.log(scale), shape
    print(f"Optimized: Mu: {mu}, Sigma: {sigma}")

    # Calculate mu sigma for alpha distributions
    shape, loc, scale = stats.lognorm.fit(np.array(alph_exps).flatten(), floc=0)
    mu, sigma = np.log(scale), shape
    print(f"Data: Mu: {mu}, Sigma: {sigma}")
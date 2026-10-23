import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    # 'paper'   : model_complexity calibration (n_cl species, set below)
    # 'test_2sp': 2-species test of model_paper/test_2sp.py
    CASE = 'test_2sp'

    DESIGN = 'base'  # test_2sp only: design as in test_2sp.py ('base', 'rep3', 'temps6', ...)

    if CASE == 'test_2sp':
        n_cl = 2
        n_media = 2
        path = 'model_paper/out/test_2sp/' if DESIGN == 'base' else f'model_paper/out/test_2sp_{DESIGN}/'
        add_name = f'_{int(n_cl)}dim_{int(n_media)}media_test'
    else:
        n_cl = 4
        n_media = 2
        relnoise = 0.1
        #path = 'model_paper/out/'#f'model_paper/out/{int(n_cl)}_dim/calibration/'#
        #path2 = path+f'noise_vs_nspecies/{int(relnoise*100)}noise/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/'
        path = f'model_paper/out/model_complexity/{int(n_cl)}_dim_{int(n_media)}media_exp_{int(relnoise*100)}noise/calibration/'
        add_name = f'_{int(n_cl)}dim_{int(n_media)}media'
    path2 = path
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
    df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')

    T_x = [0.] +[1. for _ in range(n_cl-1)]
    # 'json': final result Result_calibration<add_name>.json (after local refinement)
    # 'history': last step of the global optimization history (optimization_history1.csv)
    RESULT_SOURCE = 'json'
    print(f'Plotting {CASE}: {path}, result from ' +
          (f'Result_calibration{add_name}.json' if RESULT_SOURCE == 'json' else 'optimization_history1.csv'))
    if RESULT_SOURCE == 'json':
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'], dtype=float)
    else:
        step = 1
        optim_file2 = f"optimization_history{int(step)}.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        # Take optimal parameter values on last optimization step
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]

    # Plot resulting model
    calibr_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'output_path': path2,
        # temperature of each experiment from the column names, e.g. 'V01_M1_02C_00_MRS-mibi' -> 2.0
        'exp_temps': {c.split('_')[0]: float(c.split('_')[2].rstrip('C')) for c in data[0].columns},
        's_x': s_x,
        'media': media, 
    }
    t_model = np.linspace(0., 18., 100)
    x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    exp_clrs = {}
    i, j, k = 0, 0, 0
    try:  # paper colours (2 / 6 / 10 degC) as long as they suffice
        for exp, temp in zip(calibr_setup['exp_temps'], calibr_setup['exp_temps'].values()):
            if temp == 2.:
                exp_clrs[exp] = fm.plotting.blue_colors[i]
                i += 1
            elif temp == 6.:
                exp_clrs[exp] = fm.plotting.green_colors[j]
                j += 1
            elif temp == 10.:
                exp_clrs[exp] = fm.plotting.red_colors[k]
                k += 1
            else:
                raise IndexError
    except IndexError:  # other temperatures / more replicates: colour by temperature (cold = blue, warm = red)
        import matplotlib.colors as mcolors
        tvals = sorted(set(calibr_setup['exp_temps'].values()))
        norm = mcolors.Normalize(min(tvals) - 1., max(tvals) + 1.)
        seen = {}
        for exp, temp in sorted(calibr_setup['exp_temps'].items()):
            r = seen.get(temp, 0)
            seen[temp] = r + 1
            base = np.array(plt.cm.coolwarm(norm(temp))[:3])
            exp_clrs[exp] = tuple(base * (1. - 0.25 * r) + 0.25 * r * np.array([0.3, 0.3, 0.3]))  # replicates darker

    labels = ('Time',  r'log CFU mL$^{-1}$')
    temps_all = sorted(set(calibr_setup['exp_temps'].values()))
    for j, med in enumerate(media):
        fm.plotting.plot_all(temps_all, labels, templ_meas=fm.plotting.plot_measurements_insilico, df=data[0].filter(like=med), clrs=exp_clrs,
                 temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{med}_const_model'+add_name, time_lim=[17.5 for _ in temps_all])
    for j, med in enumerate(media):
        for temp in temps_all:
            # model rows (experiments, sorted as in calc_obs_model) at this temperature
            rows = [k for k, e in enumerate(exps) if calibr_setup['exp_temps'][e] == temp]
            obs_mibi = obs_mibi_model[rows, j, :]
            fm.plotting.plot_all([temp], labels, templ_meas=fm.plotting.plot_measurements_insilico, df=data[0].filter(like=med).filter(like=f'_{int(temp):02d}C_'),
                     clrs=exp_clrs,temps=temps_model, mtimes=t_model, mestim=obs_mibi, dir=path2, add_name=f'MiBi_{med}_const_model_{int(temp)}Grad'+add_name, time_lim=[17.5])

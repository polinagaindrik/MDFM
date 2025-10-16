import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    n_cl = 4
    n_media = 2

    path = 'model_paper/out/'
    path2 =f'model_paper/out/{int(n_cl)}_dim/calibration/'
    add_name = f'_{int(n_cl)}dim_{int(n_media)}media'
    df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
    data = [pd.read_pickle(path2+df_name) for df_name in df_names]
    data = fm.dtf.filter_dataframe_regex('V.._', data)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))

    step = 1
    optim_file2 = f"optimization_history{int(step)}.csv"
    df_optim2 = pd.read_csv(path+optim_file2)
    T_x = [0.] +[1. for _ in range(n_cl-1)]
    # Take optimal parameter values on last optimization step
    param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]

    # Plot resulting model
    calibr_setup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'output_path': path2,
        'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
        's_x': s_x,
        'media': media,
    }

    t_model = np.linspace(0., 17., 100)
    x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, t_model)
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    labels = ('Tag',  r'log CFU mL$^{-1}$')
    for j, med in enumerate(media):
        fm.plotting.plot_all([2., 10., 14.], labels, templ_meas=fm.plotting.plot_measurements_ZL2030_consttemp, df=data[0].filter(like=med),
                 temps=temps_model, mtimes=t_model, mestim=obs_mibi_model[:,j, : ], dir=path2, add_name=f'MiBi_{med}_const_model')
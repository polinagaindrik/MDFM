import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    colors_all = {
        'grey': '#808080',
        'red':'#D06062',
        'blue1': '#4E89B1',
        'purple':'#7E57A5',
        'brown':'#99582A',
        'orange_light':'#c79758',
        'yellow':'#E2B100',
        'green_dark':'#386641',
        'blue_bright':'#0982A4',
        'green_light':'#679E48',
        'orange':'#ED733E',
        'pink':'#C3568A',
    }

    n_cl = [4, 6, 8, 10]
    n_media = 2

    #x_real = []
    #x_model = []
    rms = []
    x_max_val = 1#1e9
    for n in n_cl:
        path = f'model_paper/out/{int(n)}_dim/calibration/'
        path2 = f'model_paper/out/{int(n)}_dim/calibration/'
        add_name = f'_{int(n)}dim_{int(n_media)}media'
        df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
        exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))

        # Get initial real x values from dataframe
        days_meas = [fm.dtf.get_meas_days(df_x, exp) for exp in exps]
        x_real = fm.dtf.extract_observables_from_df_x(df_x, days_meas[0], exps)

        # Calculate optimized x values
        optim_file2 = "optimization_history1.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        T_x = [1. for _ in range (n)]
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
        s_x = np.array(param_opt)[-n*n_media:].reshape((n_media, n))
        param_ode = param_opt[:-n*n_media]
        calibr_setup={
            'model': fm.mdl.fusion_model2,
            'T_x': T_x,
            'output_path': path2,
            'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
            's_x': s_x,
            'media': media, 
        }
        x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, days_meas[0])
        rms.append(root_mean_squared_error(x_real.flatten()/x_max_val, x_count.flatten()/x_max_val))
        #print(rms)

    # Now get result for 10 species for 1 different media:
    # general media

    n_media = 1
    rms_media = []
    media_names = ['gen', 'sel']
    for med in media_names:
        add_name = f'_{int(n_cl[-1])}dim_{int(n_media)}media_{med}'
        path = f'model_paper/out/{int(n_cl[-1])}_dim_{int(n_media)}media_{med}/calibration/'
        path2 = f'model_paper/out/{int(n_cl[-1])}_dim_{int(n_media)}media_{med}/calibration/'
        df_names = [f'dataframe_mibi{add_name}.pkl', f'dataframe_maldi{add_name}.pkl', f'dataframe_ngs{add_name}.pkl']
        data = [pd.read_pickle(path2+df_name) for df_name in df_names]
        media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in data[1].columns])))
        df_x = pd.read_pickle(path2+f'dataframe_x{add_name}.pkl')
        exps = sorted(list(set([s.split('_')[0] for s in df_x.columns])))
        days_meas = [fm.dtf.get_meas_days(df_x, exp) for exp in exps]
        x_real = fm.dtf.extract_observables_from_df_x(df_x, days_meas[0], exps)
        optim_file2 = "optimization_history1.csv"
        df_optim2 = pd.read_csv(path+optim_file2)
        param_opt = df_optim2.T[df_optim2.T.columns[-1]].values[1:-1]
        s_x = np.array(param_opt)[-n*n_media:].reshape((n_media, n))
        param_ode = param_opt[:-n*n_media]
        calibr_setup={
                'model': fm.mdl.fusion_model2,
                'T_x': T_x,
                'output_path': path2,
                'exp_temps': fm.output.read_from_json(''+'exp_temps_model_paper.json', dir='model_paper/'),
                's_x': s_x,
                'media': media, 
        }
        x_count, obs_mibi_model, obs_maldi_model, obs_ngsi_model, temps_model = fm.mdl.calc_obs_model(data, param_ode, calibr_setup, days_meas[0])
        rms_media.append(root_mean_squared_error(x_real.flatten()/x_max_val, x_count.flatten()/x_max_val))


    fig, ax = plt.subplots()
    ax.scatter(n_cl, np.log(rms), s=100, label='2 media', color=colors_all['blue1'])
    ax.scatter(n_cl[-1], np.log(rms_media[0]), s=100, label='general media', marker='X', color=colors_all['orange'])
    ax.scatter(n_cl[-1], np.log(rms_media[1]), s=100, label='selective media', marker='x', color=colors_all['green_light'])
    fig, ax = fm.plotting.set_labels(fig, ax, 'Number of clusters', r'log RMSE ')
    #ax.set_title('RMSE depending on number of clusters', fontsize=15)
    #ax.set_yscale('log')
    ax.legend(fontsize=15, framealpha=0)
    plt.savefig('model_paper/plot_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
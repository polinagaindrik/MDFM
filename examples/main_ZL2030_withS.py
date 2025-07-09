import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import time


if __name__ == "__main__":
    path = 'out/main_ZL2030_withS/'
    workers = 16
    cutoff0 = 0.1
    cutoff0_prop = 0.1
    add_name = '_calibr'

    # Exps wo V21 for Puten meat
    exps_ngs_all   = [f'V{i:02d}' for i in range(1, 13)] + [f'V{i:02d}' for i in range(15, 21)] + ['V22']
    exps_maldi_all = [f'V{i:02d}' for i in range(1, 20)] + ['V22'] #+ [f'V{i:02d}' for i in range(24, 28)]
    exps_calibr = [f'V{i:02d}' for i in range(1, 10)] + [f'V{i:02d}' for i in range(11, 13)] + [f'V{i:02d}' for i in range(16, 20)]

    dfs_calibr, bact_all, T_x, s_x_predefined = fm.data.prepare_ZL2030_data(exps_calibr, exps_maldi_all, exps_ngs_all, cutoff=cutoff0,
                                                                            cutoff_prop=cutoff0_prop, path=path, add_name=add_name)

    # Experiments present for each experimental method:
    exps_all   = [f'V{i:02d}' for i in range(1, 29)]
    exps_mibi_all  = [f'V{i:02d}' for i in range(1, 22)] + [f'V{i:02d}' for i in range(23, 29)]

    n_media = 2
    n_cl = np.shape(dfs_calibr[1])[0]
    print('Number of bacteria:', n_cl)
    print('T_x = ',  T_x)
    path_new = path#+f'{int(n_cl)}_dim/'
    exps_fullnames = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))

    # Fit model to df:
    calibr_presetup={
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_new,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps_calibr,
        'exp_temps': fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/'),
        }
    calibr_setup = fm.pest.define_calibr_setup_ZL2030(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)

    start = time.time()
    param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)[0]
    print((time.time()-start)/60., 'min')

    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    calibr_setup['s_x'] = s_x


    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x}, f'Result_calibration{add_name}.json', dir=path_new)
    param_opt = fm.output.read_from_json(''+f'Result_calibration{add_name}.json', dir=path_new)['param_ode']
    
    bact_all = dfs_calibr[1].T.columns
    clrs1 = {}
    for b, c in zip(bact_all, fm.plotting.colors_ngs[1:]):
        clrs1[b] = c
    fm.plotting.plot_optimization_result(np.array(param_ode), calibr_setup, np.linspace(0, 17, 100),
                                        path=path_new, clrs=clrs1, add_name='_calibration')
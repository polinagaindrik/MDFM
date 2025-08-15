import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np
import time

if __name__ == "__main__":
    path = 'model_paper/out/'
    workers = -1 
    add_name = ''

    # 'Real' model:
    temps = [2., 2., 2., 10., 10., 10., 14., 14., 14.,]
    ntr = 1
    dfs_calibr, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(fm.data.model_10sp_2media_exp, temps, ntr,
                                                                              inhib=True, noise=0.1, rel_noise=.15,
                                                                              path=path+'10_dim/', cutoff=0., cutoff_prop=0.)
    (df_mibi, df_maldi, df_ngs) = dfs_calibr
    n_cl = np.shape(dfs_calibr[1])[0]
    path_new = path+f'{int(n_cl)}_dim/calibration/'
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))
    n_media = 2

    start = time.time()
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
        'exp_temps': {exp: temp for exp, temp in zip(exps_calibr, temps)},
        }
    fm.output.json_dump(calibr_presetup['exp_temps'], 'exp_temps_model_paper.json', dir='model_paper/')
    calibr_setup = fm.pest.define_calibr_setup_ZL2030(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)[0]
    print((time.time()-start)/60., 'min')

    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    param_ode = param_opt[:-n_cl*n_media]
    calibr_setup['s_x'] = s_x

    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x}, f'Result_calibration{add_name}.json', dir=path_new)
    param_opt = fm.output.read_from_json(''+f'Result_calibration{add_name}.json', dir=path_new)['param_ode']
import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm

import pandas as pd
import numpy as np


def prepare_insilico_data(insilico_model, n_cl, temps, ntr, S_matrix_setup, param_ode=None, x10=None, path='', inhib=False, noise=0., rel_noise=.0, cutoff=0., cutoff_prop=0., add_name='', exp_start_offset=0):
    data = insilico_model(n_cl, temps, ntr, S_matrix_setup, param_ode=param_ode, path=path, inhib=inhib, noise=noise, rel_noise=rel_noise, x10=x10, add_name=add_name, exp_start_offset=exp_start_offset)
    dfs = data[:-1]
    (df_mibi, df_maldi, df_ngs) = dfs
    df_ngs, bact_ngs = fm.dtf.preprocess_dataframe(df_ngs, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    df_maldi, bact_maldi = fm.dtf.preprocess_dataframe(df_maldi, cutoff=cutoff, cutoff_prop=cutoff_prop, calc_prop=False)
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    # Take union of ngs and maldi bacteria
    df_maldi, df_ngs, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(df_maldi, df_ngs, cutoff=0.001, media=media)
    bact_all = sorted(set(list(bact_maldi) + list(bact_ngs)))
    fm.data.save_all_dfs([df_mibi, df_maldi, df_ngs], names=[f'mibi{add_name}_preprocessed', f'maldi{add_name}_preprocessed', f'ngs{add_name}_preprocessed'], path=path)
    return [df_mibi, df_maldi, df_ngs], bact_all, T_x, s_x_predefined

def model_2media_gensel(n_cl, temps, ntr, S_matrix_setup, **kwargs):
    s_x = np.array(S_matrix_setup["s_x"]).flatten()
    T_x = np.array(S_matrix_setup["T_x"])
    media = ['sel1', 'gen1']
    return model_wotemp(n_cl, temps, ntr, s_x, T_x, media, **kwargs)

def model_wotemp(n_cl, temps, ntr, s_x, T_x, media, param_ode=None, x10=None, path='', inhib=False, noise=0., rel_noise=0., add_name='', exp_start_offset=0):
    np.random.seed(46987)
    t = np.linspace(0., 17., 18)
    if param_ode is None:
       print('No parameter vector provided.')
       exit()
    if not inhib:
        param_ode[-n_cl*n_cl:] = 0.
    param_model = np.concatenate([param_ode, s_x, T_x])
    if x10 is not None:
        x0 = fm.data.set_initial_vals(x10, temps, n_cl)
    else:
        x10, x0 = fm.data.get_random_initial_vals(temps, n_cl)
    df_mibi, df_maldi, df_ngs, df_realx, _ = fm.data.generate_data_dfs(fm.mdl.fusion_model_distr, t, param_model, x0, temps, n_cl, n_traj=ntr, noise=noise, rel_noise=rel_noise, media=media, exp_start_offset=exp_start_offset)
    fm.data.save_all_dfs([df_mibi, df_maldi, df_ngs, df_realx], names=[f'mibi{add_name}', f'maldi{add_name}', f'ngs{add_name}', f'x{add_name}'], path=path)
    print(add_name, param_ode, '\n')
    fm.data.json_dump({'param_ode': [x00  for i in range (len(temps)) for x00 in x10[i]]+list(param_ode), 's_x': s_x, 'T_x': T_x}, f'Result_real_paramdistrib{add_name}.json', dir=path)
    return df_mibi, df_maldi, df_ngs, df_realx


def data_generation_distribution(n_exps, n_cl, param_ode_templ, s_x, path=''):
    dfs_mibi, dfs_maldi, dfs_ngs = [], [], []
    for i in range (n_exps):
        np.random.seed(46987*(i+1))
        add_name = f'_{i}'
        alph_sampled = np.random.lognormal(mean=0., sigma=0.5, size=n_cl)
        param_ode_templ[n_cl:n_cl*2] = alph_sampled
        param_init = param_ode_templ

        temps = [2.,]
        ntr = 1
        x10 = np.array([[2.76, 1.8]])
        S_matrix_setup = {
            "s_x": s_x,
            "T_x": np.array([1. for _ in range (n_cl)]),
        }
        dfs_calibr, bact_all, T_x, s_x_predefined = prepare_insilico_data(model_2media_gensel, n_cl, temps, ntr, S_matrix_setup, param_ode=param_init, x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=path, add_name=add_name, exp_start_offset=i*ntr)
        (df_mibi, df_maldi, df_ngs) = dfs_calibr
        dfs_mibi.append(df_mibi)
        dfs_maldi.append(df_maldi)
        dfs_ngs.append(df_ngs)
    return fm.dtf.merge_dfs(dfs_mibi), fm.dtf.merge_dfs(dfs_maldi), fm.dtf.merge_dfs(dfs_ngs)

def data_calibration_distribution(dfs, path=''):
    (df_mibi, df_maldi, df_ngs) = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs[0].columns])))

    calibr_presetup={
            'model': fm.mdl.fusion_model_distr,
            'T_x': T_x,
            'workers': workers, # number of threads for multiprocessing
            'output_path': path_new,
            'n_cl': n_cl,
            'n_media': n_media,
            'dfs': dfs,
            'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
            'exps': exps_calibr,
            'exp_temps': {exp: temp for exp, temp in zip(exps_calibr, temps)},
            'media': media, 
        }
    fm.output.json_dump(calibr_presetup['exp_temps'], 'exp_temps_model_paper.json', dir=path)

    x0_bnds_all = tuple([(1., 4.5) for _ in range (calibr_presetup['n_cl']) for _ in range (len(calibr_presetup['exps']))])
    inhib_bnds = [(0.02, 3.) for _ in range (n_cl*(n_cl-1))]
    param_ode_bnds = tuple([(1e-6, 1e-2) for _ in range (n_cl)] + # lambd
                            [(.01,  4.) for _ in range (n_cl) for _ in range (len(calibr_presetup['exps']))]  + # alph
                            [(6., 12.)]                         + # N_max
                            inhib_bnds)
    S_bnds = tuple([(0.01, 1.) for _ in range (calibr_presetup['n_cl']*n_media)])
    calibr_setup = calibr_presetup
    calibr_setup['param_bnds'] =  x0_bnds_all + param_ode_bnds + S_bnds

    print('Start optimization...')
    param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS_withdiffalpha, calibr_setup)[0]
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
    calibr_setup['s_x'] = s_x
    fm.output.json_dump({'param_ode': param_opt[:-n_cl*n_media].astype(list), 's_x': s_x.flatten(), 'T_x': T_x}, 'Result_calibration.json', dir=path_new)
    return param_opt, calibr_setup


def estimate_parameter_set(param_init, s_x, n_cl, add_name='',path=''):
    temps = [2.,]
    ntr = 1
    x10 = np.array([[2.76, 1.8]])
    S_matrix_setup = {
        "s_x": s_x,
        "T_x": np.array([1. for _ in range (n_cl)]),
    }
    dfs_calibr, bact_all, T_x, s_x_predefined = prepare_insilico_data(model_2media_gensel, n_cl, temps, ntr, S_matrix_setup, param_ode=param_init, x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=path, add_name=add_name)

    (df_mibi, df_maldi, df_ngs) = dfs_calibr
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps_calibr = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))

    calibr_presetup={
        'model': fm.mdl.fusion_model_distr,
        'T_x': T_x,
        'workers': workers, # number of threads for multiprocessing
        'output_path': path_new,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps_calibr,
        'exp_temps': {exp: temp for exp, temp in zip(exps_calibr, temps)},
        'media': media, 
        }
    fm.output.json_dump(calibr_presetup['exp_temps'], 'exp_temps_model_paper.json', dir='model_paper/')

    x0_bnds_all = tuple([(1., 4.5) for _ in range (calibr_presetup['n_cl']) for _ in range (len(calibr_presetup['exps']))])
    #inhib_bnds = [(0.02, 3.) if i!=j else (0., 0.)
    #              for i in range (n_cl)
    #              for j in range (n_cl)]
    inhib_bnds = [(0.02, 3.) for _ in range (n_cl*(n_cl-1))]
    param_ode_bnds = tuple([(1e-6, 1e-2) for _ in range (n_cl)] + # lambd
                           [(.01,  4.) for _ in range (n_cl)]   +  # alph
                           [(6., 12.)]                         + # N_max
                            inhib_bnds)
    S_bnds = tuple([(0.01, 1.) for _ in range (calibr_presetup['n_cl']*n_media)])

    calibr_setup = calibr_presetup
    calibr_setup['param_bnds'] =  x0_bnds_all + param_ode_bnds + S_bnds
    param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)[0]
    return param_opt, calibr_setup

if __name__ == "__main__":
    path = 'out/'
    workers = -1
    n_cl = 2
    n_media = 2
    relnoise = 0.1

    add_name = ''
    path_new = path+'main_param_distrib2/'
    # 'Real' model:
    n_exps = 50
    temps = [2. for _ in range (n_exps)]
    ntr = 1

    # Define parameter vector
    #x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=path)['x0'])[:len(temps), :n_cl]
    #S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=path)
    T_x = np.array([1., 1.])
    s_x = np.array([[0.1, 0.95], [0.7, 0.6 ]])
    x0 = np.array([5.8e2, 6.5e1, 1., 1., 1.])
    param_ode = np.array([  8e-4,   5e-5,   # 3e-3,
                          np.nan, np.nan,   # 1.2,
                           7.,              # identifiable at 1.3e7
                          2.5e-8, 9.5e-8])
    
    dfs_calibr = data_generation_distribution(n_exps, n_cl, param_ode, s_x, path=path_new)
    param_opt, calibr_setup = data_calibration_distribution(dfs_calibr, path=path_new)

    '''
    for i in range (n_exps):
        add_name = f'_{i}'
        alph_sampled = np.random.lognormal(mean=0., sigma=0.5, size=n_cl)
        param_ode[n_cl:n_cl*2] = alph_sampled
        param_init = param_ode
        param_opt, calibr_setup = estimate_parameter_set(param_init, s_x, n_cl, add_name=add_name, path=path_new)

        s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
        calibr_setup['s_x'] = s_x

        fm.output.json_dump({'param_ode': param_opt[:-n_cl*n_media].astype(list), 's_x': s_x.flatten(), 'T_x': T_x}, f'Result_calibration{add_name}.json', dir=path_new)
        #param_opt = fm.output.read_from_json(''+f'Result_calibration{add_name}.json', dir=path_new)['param_ode']

        # TODO: Consider: Now we estimate the whole parameter set for each run, but only alpha changes. Maybe estimate all exps together but with multiple alpha params.
        # e.g param_estim = (x0, alph_1_exp1, alph_2_exp1, ..., alph_1_exp2, alph_2_exp2, ... , param_ode_other)
    '''
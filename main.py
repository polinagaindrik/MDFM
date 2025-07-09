import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import numpy as np


if __name__ == "__main__":
    # ZL2030 data:
    #exps_calibr = [f'V{i:02d}' for i in range(1, 10)]
    #exps_mibi, exps_maldi, exps_ngs = [exps_calibr for _ in range (3)]
    #dfs_calibr, bact, T, _ = fm.data.prepare_ZL2030_data(exps_mibi, exps_maldi, exps_ngs, cutoff=0.05, cutoff_prop=0.)

    # In-silico data:
    temps = [2., 10., 14.]
    dfs_calibr, bact, T, _ = fm.data.prepare_insilico_data(fm.data.model_4sp_2media_inhib, temps, 1)

    n_media = 2
    n_cl = np.shape(dfs_calibr[1])[0]
    exps = sorted(list(set([s.split('_')[0] for s in dfs_calibr[0].columns])))

    ######################################## Calibration #################################################
    #Example of the parameter bounds for  fm.mdl.fusion_model_linear:
    x0_bnds_all = tuple([(1., 4.5) for _ in range (n_cl) for _ in range (len(exps))])
    S_bnds = tuple([(0.05, 1.) for _ in range (n_cl*n_media)])
    inhib_bnds = tuple([(0., 0.) for _ in range (n_cl*n_cl)])
    #inhib_bnds = tuple([(0.02, 3.) if i!=j else (0., 0.)
    #              for i in range (n_cl)
    #              for j in range (n_cl)])
    calibr_setup={
        'model': fm.mdl.fusion_model_linear,
        'param_bnds': x0_bnds_all +                              # Initial values x bounds
                      tuple([(1e-6, .01) for _ in range (n_cl)] + # ODE parameter bounds
                            [(.1,  1.)  for _ in range (n_cl)] +
                            [(.1,  1.)  for _ in range (n_cl)] +
                            [(6., 9.)]) + inhib_bnds + S_bnds,
        'T_x': T,
        'workers': -1, 
        'output_path': 'out/',
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps,
        'exp_temps': {exp: temp for exp, temp in zip(exps, temps)},
    }

    param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)[0]

    # Here parameters for S matrix and ODEs:
    s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl)) 
    param_ode = np.array(param_opt)[n_cl*len(exps):-n_cl*n_media]
    
    # Save calibration result:
    fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T}, 'Result_calibration.json')

    ####################################### Prediction ###################################################3
    dfs_predict, _, _, _ = fm.data.prepare_insilico_data(fm.data.model_4sp_2media_inhib, [5.], 1)
    x0_bnds = tuple([(1., 4.5) for _ in range (n_cl)])
    exps_predict = sorted(list(set([s.split('_')[0] for s in dfs_predict[0].columns])))

    # Define prediction setup
    prediction_setup = calibr_setup
    prediction_setup['param_ode'] = param_ode
    prediction_setup['dfs'] = dfs_predict
    prediction_setup['param_bnds'] = x0_bnds
    prediction_setup['exps'] = exps_predict
    prediction_setup['s_x'] = s_x
    prediction_setup['exp_temps'] = {'V01': 5.}

    # Run initial value estimation for prediction:
    x0_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals, prediction_setup)[0]

    # Save optimized initial values for each experiment
    fm.output.save_values_each_experiment(x0_opt, exps_predict, n_cl, filename='Initial_values')
    
    # Read fom output file:
    x0_opt = fm.output.read_from_json('Initial_values.json')

    # Plot the prediction curves:
    fm.plotting.plot_prediction_result(prediction_setup['param_ode'], x0_opt,  prediction_setup, np.linspace(0, 17, 100))
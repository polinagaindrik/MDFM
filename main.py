"""
Minimal example of fusion_model: calibration and prediction on in-silico data
=============================================================================

Self-contained, no setup files or measurement data needed:

  1) in-silico data: model fusion_model2 with known parameters, 2 species, 2 media (sel1, gen0),
     3 experiments at 2, 6 and 10 degC
  2) calibration: initial values, ODE parameters and media coefficients S together (cost_withS)
  3) prediction: new experiment at 8 degC, only its initial values are estimated (cost_initvals)

Results and figures are written to out/main_example/. The figures need a LaTeX installation.
Run from any folder:
    python main.py
"""
import os

import numpy as np

import fusion_model as fm

PATH = 'out/main_example/'
WORKERS = -1    # processes of differential evolution, -1 = all cores
MAXITER = 200   # max. generations of the calibration; it stops earlier when converged

N_CL = 2
TEMPS_CALIBR = [2., 6., 10.]
TEMPS_PREDICT = [8.]

# "True" parameters of the in-silico data
PARAM_ODE_TRUE = np.array([-3.6, -4.8,          # lambda_1 (log10)
                           1.7, 1.9,            # lambda_exp
                           0.8, 0.8,            # alpha_0
                           0.4, 0.8,            # alpha_exp
                           9.9, 0.7,            # N_1 (log10), N_exp
                           0.0, 0.5, 1.9, 0.0]) # k_ij (N_CL x N_CL)
X10_CALIBR = [[1.8, 3.3], [2.0, 3.3], [3.0, 2.8]]  # log10 initial values, experiments x species
X10_PREDICT = [[2.3, 2.6]]
S_MATRIX = {'s_selective': [[0.87, 0.09]],  # media coefficients of the selective medium sel1
            's_general0': [0.7, 0.56],      # and of the general medium gen0
            'T_x': [1., 1.]}


def generate_data(temps, x10, add_name):
    """In-silico plate counts, MALDI and NGS data frames with 10 % relative noise."""
    return fm.data.prepare_insilico_data(fm.data.model_2media_expgensel, N_CL, temps, 1, S_MATRIX,
                                         param_ode=PARAM_ODE_TRUE.copy(), x10=x10, path=PATH, inhib=True,
                                         rel_noise=0.1, add_name=add_name)


def experiments(df_platecount):
    """Names of the experiments in the data frame columns, e.g. ['V01', 'V02', 'V03']."""
    return sorted(set(col.split('_')[0] for col in df_platecount.columns))


if __name__ == "__main__":
    os.makedirs(PATH, exist_ok=True)
    # Optimization histories in the folder of this run (default: out/)
    fm.pest.optimization.output_file = PATH + 'optimization_history1.csv'
    fm.pest.optimization.output_file_predict = PATH + 'optimization_history_predict.csv'
    t_plot = np.linspace(0., 17., 100)


    ######################################## Calibration #################################################
    dfs_calibr, bact, T_x, s_x_predefined = generate_data(TEMPS_CALIBR, X10_CALIBR, '_calibr')
    exps = experiments(dfs_calibr[0])
    media = sorted(set(col.split('_')[-1].split('-')[0] for col in dfs_calibr[1].columns))
    n_media = len(media)

    calibr_setup = fm.pest.define_calibr_setup_insilico({
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': WORKERS,
        'output_path': PATH,
        'n_cl': N_CL,
        'n_media': n_media,
        'media': media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps,
        'exp_temps': dict(zip(exps, TEMPS_CALIBR)),
    }, inhib=True, s_x_predefined=s_x_predefined)

    param_opt, cost_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup, maxiter=MAXITER)

    # Parameter vector: [ x0 (N_CL per experiment) | ODE parameters | S (n_media x N_CL) ]
    param_ode = param_opt[N_CL*len(exps):-N_CL*n_media]
    s_x = param_opt[-N_CL*n_media:].reshape((n_media, N_CL))
    print(f'Calibration: cost = {cost_opt:.4g}')
    print('ODE parameters, true:     ', np.round(PARAM_ODE_TRUE, 2))
    print('ODE parameters, estimated:', np.round(param_ode, 2))
    print(f'S, estimated (rows {media}):', np.round(s_x, 2).tolist())

    fm.output.json_dump({'param_ode': param_opt, 's_x': s_x, 'T_x': T_x, 'cost': cost_opt},
                        'Result_calibration.json', dir=PATH)
    fm.plotting.plot_optimization_result(param_opt[:-N_CL*n_media], dict(calibr_setup, s_x=s_x), t_plot, path=PATH,
                                         add_name='_calibration')

    ######################################## Prediction ##################################################
    dfs_predict = generate_data(TEMPS_PREDICT, X10_PREDICT, '_predict')[0]
    exps_predict = experiments(dfs_predict[0])

    prediction_setup = dict(calibr_setup,  # copy: calibr_setup stays unchanged
                            param_ode=param_ode,
                            s_x=s_x,
                            dfs=dfs_predict,
                            param_bnds=tuple((0., 5.) for _ in range(N_CL*len(exps_predict))),
                            exps=exps_predict,
                            exp_temps=dict(zip(exps_predict, TEMPS_PREDICT)))

    # Only the initial values of the new experiment are estimated
    x0_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals, prediction_setup)[0]
    fm.output.save_values_each_experiment(x0_opt, exps_predict, N_CL, dir=PATH, filename='Initial_values_prediction')
    print('Initial values (log10), true:     ', X10_PREDICT[0])
    print('Initial values (log10), estimated:', np.round(x0_opt, 2))

    x0_dict = fm.output.read_from_json('Initial_values_prediction.json', dir=PATH)
    fm.plotting.plot_prediction_result(param_ode, x0_dict, prediction_setup, t_plot, path=PATH, add_name='_prediction')
    print(f'Results and figures in {PATH}')

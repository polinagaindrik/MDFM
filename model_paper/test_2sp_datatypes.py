"""
Test case: 2-species model, calibration with only two of the three data types
=============================================================================

Same problem and pipeline as test_2sp.py (2 species, 2 media, 3 experiments at 2, 6 and
10 degC, 10% relative noise), but the cost uses only two data types per scenario:

  'ngs_maldi'  : NGS + MALDI           (no plate counts)
  'ngs_mibi'   : NGS + plate counts (MiBi, PC)    (no MALDI)
  'maldi_mibi' : MALDI + plate counts (MiBi, PC)  (no NGS)

For each scenario:
  1) global optimization (differential evolution, max. DE_MAXITER generations)
                                                         -> <scenario>/Result_calibration<add_name>_global.json
  2) local refinement (L-BFGS-B multistart)              -> <scenario>/Result_calibration<add_name>.json
  3) profile likelihood (local fits; objective see OBJECTIVE) -> <scenario>/profile_likelihood/...

The in-silico data are those of test_2sp.py (model_paper/out/test_2sp/, generated if missing).
The omitted data type is still simulated but has no weight in the cost / profile objective.
Each stage can be switched off (RUN_*) and is then read from its saved result. Run from the
repository root:
    python model_paper/test_2sp_datatypes.py
"""
import os
import sys
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import make_param_names, exp_temps_from_columns
from test_2sp import N_CL, PARAM_ODE_TRUE, true_parameter_vector, print_comparison

import numpy as np


SCENARIOS_ALL = {
    'ngs_maldi': ['ngs', 'maldi'],
    'ngs_mibi': ['ngs', 'mibi'],
    'maldi_mibi': ['maldi', 'mibi'],
}


def profile_objective(objective, data_array, use):
    """-2 log L profile objective that ignores the data types not in use (their n_k set to 0)."""
    n_data = pl.make_gaussian_neg2loglik(data_array).n_data
    n_data = [n if k in use else 0 for n, k in zip(n_data, fm.pest.DATA_TYPES)]
    return {'neg2logL': pl.GaussianNeg2LogLik, 'geomean': pl.GeometricMeanNeg2LogLik}[objective](n_data)


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    data_path = 'model_paper/out/test_2sp/'            # in-silico data (shared with test_2sp.py)
    out_root = 'model_paper/out/test_2sp_datatypes/'   # one subfolder per scenario
    SCENARIOS = ['ngs_maldi', 'ngs_mibi', 'maldi_mibi']  # subset of SCENARIOS_ALL, run one after the other
    n_cl = N_CL
    n_media = 2
    relnoise = 0.1
    temps = [2., 6., 10.]          # 3 experiments, 3 temperatures
    ntr = 1
    add_name = f'_{n_cl}dim_{n_media}media_test'
    workers = 20                   # differential evolution: -1 = all cores
    n_jobs = 20                    # local multistart / profile likelihood processes

    RUN_DATA = False               # generate the in-silico data (also done if no data in data_path)
    RUN_GLOBAL = True              # 1) differential evolution
    RUN_LOCAL = True               # 2) local refinement
    RUN_PROFILE = True             # 3) profile likelihood

    DE_MAXITER = 700               # max. differential evolution generations (1000 in test_2sp.py)
    # local step and profile fits: 'L-BFGS-B' or 'least_squares' (see test_2sp.py)
    LOCAL_METHOD = 'L-BFGS-B'

    # profile likelihood settings (see test_2sp.py)
    PROFILE = 'all'                # 'x0', 'ode', 'ode+S' or 'all' (= x0 + ode + S)
    FIX_X0 = False                 # True: x0 held at the estimate
    FIX_S = False                  # False (re-fitted), 'estimate' or 'true' (held at the true values)
    # 'neg2logL' : -2 log L, data types weighted by their number of points
    # 'geomean'  : -2 log L form of the calibration cost (equal weight per data type)
    # (both only over the data types of the scenario)
    OBJECTIVE = 'geomean'
    SHOW_TRUE_VALUE = True         # False: profile plots without the green "true value" line
    # ================================================================

    if OBJECTIVE not in ('neg2logL', 'geomean'):
        raise ValueError("OBJECTIVE must be 'neg2logL' or 'geomean'")
    if FIX_X0 and PROFILE in ('x0', 'all'):
        raise ValueError('PROFILE=' + repr(PROFILE) + ' profiles x0, so FIX_X0 must be False')
    if FIX_S not in (False, 'estimate', 'true'):
        raise ValueError("FIX_S must be False, 'estimate' or 'true'")
    os.makedirs(data_path, exist_ok=True)
    start = time.time()

    # in-silico data (as in test_2sp.py) ---------------------------------
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=fm.data.SETUP_DIR)
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=fm.data.SETUP_DIR)['x0'])[:len(temps), :n_cl]
    if RUN_DATA or not os.path.exists(data_path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup, param_ode=PARAM_ODE_TRUE.copy(),
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=data_path,
            add_name=add_name)
    else:
        import pandas as pd
        dfs = [pd.read_pickle(data_path + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
        _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
            dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    df_mibi, df_maldi, df_ngs = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    bact = list(df_maldi.index)
    print(f'Data: {len(exps)} experiments {exps}, media {media}, species {bact}')

    calibr_presetup = {
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(df_mibi),
        'media': media,
    }
    names = make_param_names(exps, bact, media)
    param_true = true_parameter_vector(data_path, add_name, media, n_cl)

    for scenario in SCENARIOS:
        use = SCENARIOS_ALL[scenario]
        path = out_root + scenario + '/'
        os.makedirs(path, exist_ok=True)
        print(f'\n{"=" * 70}\nScenario {scenario}: data types {use}\n{"=" * 70}')

        # calibration setup: cost over the data types in use only --------
        calibr_presetup_sc = dict(calibr_presetup, output_path=path,
                                  aggregation_func=fm.pest.SelectedDataTypes(fm.pest.cost_sum_and_geometric_mean, use))
        calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup_sc, inhib=True,
                                                            s_x_predefined=s_x_predefined, s_x=None)
        calibr_setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
        assert len(param_true) == len(calibr_setup['param_bnds']) == len(names)
        print(f'Cost (calibration objective) at the true parameters: '
              f'{fm.pest.cost_withS(param_true, calibr_setup, None):.6g}')

        # 1) global optimization -----------------------------------------
        fm.pest.optimization.output_file = path + 'optimization_history1.csv'
        if RUN_GLOBAL:
            param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup,
                                                                   maxiter=DE_MAXITER)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            s_x_glob = np.array(param_glob)[-n_cl*n_media:].reshape((n_media, n_cl))
            fm.output.json_dump({'param_ode': param_glob.astype(list), 's_x': s_x_glob, 'T_x': T_x, 'cost': cost_glob,
                                 'data_types': use},
                                f'Result_calibration{add_name}_global.json', dir=path)
        else:
            param_glob = np.array(fm.output.read_from_json(f'Result_calibration{add_name}_global.json', dir=path)['param_ode'])
        print_comparison(names, param_glob, param_true, calibr_setup['param_bnds'],
                         f'[{scenario}] Global optimization vs. truth:')

        # 2) local refinement --------------------------------------------
        if RUN_LOCAL:
            param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup,
                                                             maxiter=200 if LOCAL_METHOD == 'least_squares' else 1000,
                                                             n_restarts=3, jitter_frac=0.05, n_jobs=n_jobs,
                                                             method=LOCAL_METHOD,
                                                             history_file=path + 'optimization_history_local.csv')
            print(f'{(time.time() - start) / 60.:.1f} min (global + local), cost = {cost_opt:.6g}')
            s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
            fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                                 'data_types': use},
                                f'Result_calibration{add_name}.json', dir=path)
        else:
            param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'])
        print_comparison(names, param_opt, param_true, calibr_setup['param_bnds'],
                         f'[{scenario}] Local refinement vs. truth:')

        # 3) profile likelihood ------------------------------------------
        if RUN_PROFILE:
            out_path = path + 'profile_likelihood/'
            os.makedirs(out_path, exist_ok=True)
            pl_setup = dict(calibr_setup)
            pl_setup['aggregation_func'] = profile_objective(OBJECTIVE, pl_setup['data_array'], use)
            print(f'Profile objective ({OBJECTIVE}) at the truth: {fm.pest.cost_withS(param_true, pl_setup, None):.6g}, '
                  f'at the estimate: {fm.pest.cost_withS(param_opt, pl_setup, None):.6g}')

            n_x0 = n_cl * len(exps)
            n_ode = 4 * n_cl + 2 + n_cl * n_cl
            idx_x0 = list(range(n_x0))
            idx_ode = list(range(n_x0, n_x0 + n_ode))
            idx_S = list(range(n_x0 + n_ode, len(param_opt)))
            profile_indices = {'x0': idx_x0, 'ode': idx_ode, 'ode+S': idx_ode + idx_S, 'all': None}[PROFILE]
            fixed_indices = (idx_x0 if FIX_X0 else []) + (idx_S if FIX_S else [])
            if FIX_S == 'true':  # S at the true values, the other parameters are re-fitted (refine=True)
                param_opt = np.array(param_opt, dtype=float)
                param_opt[idx_S] = param_true[idx_S]
            if FIX_S:
                print(f'S held fixed at the {FIX_S} values: {np.round(np.asarray(param_opt)[idx_S], 4)}')
            if profile_indices is None:
                profile_indices = list(range(len(param_opt)))
            profile_indices = [i for i in profile_indices if i not in fixed_indices]
            tag = (f"{add_name}_{scenario}_{PROFILE.replace('+', '')}_{OBJECTIVE}" + ('_fixx0' if FIX_X0 else '')
                   + {False: '', 'estimate': '_fixS', 'true': '_fixStrue'}[FIX_S])
            out_csv = out_path + f'profile_likelihood_results{tag}.csv'

            profile_options = ({'method': 'least_squares', 'maxiter': 200} if LOCAL_METHOD == 'least_squares'
                               else {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            df, cis = pl.run_profile_likelihood_all(
                fm.pest.cost_withS, param_opt, pl_setup,
                span=3, n_points=12, method='local',
                n_jobs=n_jobs,
                out_csv=out_csv,
                plot_path=out_path + f'profile_likelihood{tag}.png',
                param_names=names,
                n_restarts=1, jitter_frac=0.05,
                profile_indices=profile_indices,
                fixed_indices=fixed_indices,
                refine=FIX_S == 'true',        # re-fit first when S is moved to the true values
                stop_factor=3,
                minimize_options=profile_options,
                true_params=param_true if SHOW_TRUE_VALUE else None,
                verbose=1,
                restart_on_better=False,
                resume=True,
            )
            pl.plot_profile_likelihood_individual(out_csv, out_dir=out_path + 'individual', file_ext='pdf',
                                                  true_params=param_true if SHOW_TRUE_VALUE else None)
        print(f'{(time.time() - start) / 60.:.1f} min in total (after scenario {scenario})')

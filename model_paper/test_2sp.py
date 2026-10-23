"""
Test case: 2-species model, 3 experiments at 3 temperatures
===========================================================

Same pipeline as main_paper_calibr.py + profile_likelihood_model_paper.py on a
small problem, to check parameter estimation and identifiability end to end:

  1) in-silico data: fusion_model2 with known ("true") parameters, 2 species,
     2 media (sel1, gen1), 3 experiments at 2, 6 and 10 degC, 10% relative noise
  2) global optimization (differential evolution)       -> Result_calibration<add_name>_global.json
  3) local refinement (L-BFGS-B multistart)              -> Result_calibration<add_name>.json
  4) profile likelihood (local fits; objective see OBJECTIVE) -> profile_likelihood/...

Each stage can be switched off (RUN_*) and is then read from its saved result,
e.g. to re-run only the profiles. Run from the repository root:
    python model_paper/test_2sp.py
"""
import os
import sys
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import (make_param_names, exp_temps_from_columns, resolve_fixed_values,
                                            fixed_values_from_result, print_parameter_table)

import numpy as np


# ----------------------------------------------------------------------
# True model (2 species; values of species 00 and 01 of the 4-species in-silico model)
# ----------------------------------------------------------------------
N_CL = 2
PARAM_ODE_TRUE = np.array(
    [-3.6, -4.8] +          # lambda_1   (log10)
    [1.7, 1.9] +            # lambda_exp
    [0.8, 0.8] +            # alpha_0
    [0.4, 0.8] +            # alpha_exp
    [9.9, 0.7] +            # N_1 (log10), N_exp
    [0.0, 0.5,              # k_ij (row i: inhibition of species i by species j; k_ii = 0)
     1.9, 0.0]
)
GEN_MEDIA = ['sel1', 'gen1']  # media order of fm.data.model_2media_expgensel (S rows in the true file)


def true_parameter_vector(path, add_name, media_sorted, n_cl):
    """True parameters in the layout of the calibration vector [x0 | ODE | S (media sorted)]."""
    tr = fm.output.read_from_json(f'Result_real{add_name}.json', dir=path)
    s_true = np.array(tr['s_x'], dtype=float).reshape(len(GEN_MEDIA), n_cl)
    s_true = np.array([s_true[GEN_MEDIA.index(m)] for m in media_sorted])  # calibration uses sorted media
    return np.concatenate([np.ravel(tr['param_ode']), s_true.ravel()])


def print_comparison(names, estimate, true, bnds, title):
    print(f'\n{title}')
    print(f'{"parameter":<22}{"true":>10}{"estimate":>12}{"bounds":>18}')
    for n, t, e, b in zip(names, true, estimate, bnds):
        if b[1] > b[0]:
            flag = '  <- at bound' if np.isclose(e, b[0]) or np.isclose(e, b[1]) else ''
            print(f'{n:<22}{t:>10.4g}{e:>12.4g}   ({b[0]:g}, {b[1]:g}){flag}')


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    path = 'model_paper/out/test_2sp/'
    n_cl = N_CL
    n_media = 2
    relnoise = 0.1
    temps = [2., 6., 10.]          # 3 experiments, 3 temperatures
    ntr = 1
    add_name = f'_{n_cl}dim_{n_media}media_test'
    workers = 3#-1                   # differential evolution: -1 = all cores
    n_jobs = 3#os.cpu_count()-2 or 1   # local multistart / profile likelihood processes

    RUN_DATA = False               # generate the in-silico data
    RUN_GLOBAL = False             # 2) differential evolution
    RUN_LOCAL = True               # 3) local refinement
    RUN_PROFILE = False             # 4) profile likelihood

    # optimizer settings
    # global step: 'de'            differential evolution (as before)
    #              'multistart'    N_STARTS local fits from Latin hypercube points (recommended for many species)
    #              'de+multistart' differential evolution, then the multistart incl. the DE result
    GLOBAL_METHOD = 'de'
    N_STARTS = 1
    # local step and profile fits: 'L-BFGS-B' (as before) or 'least_squares' (trust-region least squares on
    # the residuals, parameters scaled, much faster convergence for ill-conditioned problems)
    LOCAL_METHOD = 'least_squares'
    # local step: parameters held at a fixed value, only the others are refined (e.g. non-identifiable ones
    # set to 0). Keys: index or name (see the printed parameter table), e.g. {17: 0., 18: 0.} or
    # {'k_00,01': 0.}; the value may lie outside the bounds. Stored as 'fixed_params' in
    # Result_calibration*.json and kept fixed in the profile likelihood (not profiled, not re-fitted).
    FIX_PARAMS = {'k_00,01': 0.,
                  'alpha_exp,00': 0.,
                  'lambda_exp,00': 0.}
    # True: FIX_PARAMS also in the global step (new differential evolution / multistart search with these
    # parameters fixed) instead of only refining from the earlier global result
    FIX_IN_GLOBAL = True

    # profile likelihood settings
    PROFILE = 'all'              # 'x0', 'ode', 'ode+S' or 'all' (= x0 + ode + S)
    FIX_X0 = False                 # True: x0 held at the estimate (faster, profiles conditional on x0)
    # S (media matrix): False = re-fitted at every profile point (default)
    #                   'estimate' = held at the estimated values, 'true' = held at the true values (in-silico)
    FIX_S = False        # S re-fitted at every profile point (as before)
    #FIX_S = 'estimate'   # S held at the estimated values
    #FIX_S = 'true'       # S held at the true values (in-silico data only)  
    # 'neg2logL'   : -2 log L, data types weighted by their number of points
    # 'geomean'    : -2 log L form of the calibration cost cost_sum_and_geometric_mean
    #                (same optimum as the calibration, equal weight per data type)
    # 'calibration': cost_sum_and_geometric_mean itself (scale linearized: G_opt / N)
    OBJECTIVE = 'geomean'
    SHOW_TRUE_VALUE = True         # False: profile plots without the green "true value" line
    # Add points to the existing profiles (same PROFILE / FIX_* / OBJECTIVE -> same CSV) instead of a new run:
    #   None   : normal profile run
    #   'auto' : every profile side that does not reach the threshold (and does not end at the bound)
    #            gets EXTEND_N_POINTS new points up to the bound; a side stops once Delta > 3 x threshold
    #   dict   : {index: spec} with spec = {'upper': value, 'lower': value, 'n_points': n} (new points from
    #            the current end of the profile to value) or a list of values (anywhere, also to fill gaps);
    #            add 'auto': True to also extend all other parameters automatically.
    #            Indices are printed as "Parameter indices" when EXTEND is set.
    #   e.g. EXTEND = {2: {'upper': 5.0, 'n_points': 6}, 12: [0.01, 0.03, 0.1, 0.3, 0.6], 'auto': True}
    EXTEND = None
    EXTEND_N_POINTS = 6
    # ================================================================

    os.makedirs(path, exist_ok=True)
    start = time.time()

    # 1) in-silico data -------------------------------------------------
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir='model_paper/out/')
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir='model_paper/out/')['x0'])[:len(temps), :n_cl]
    if RUN_DATA or not os.path.exists(path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup, param_ode=PARAM_ODE_TRUE.copy(),
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=path, add_name=add_name)
    else:
        import pandas as pd
        dfs = [pd.read_pickle(path + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
        _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
            dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    df_mibi, df_maldi, df_ngs = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    bact = list(df_maldi.index)
    print(f'Data: {len(exps)} experiments {exps}, media {media}, species {bact}')

    # calibration setup (as in main_paper_calibr.py) ---------------------
    calibr_presetup = {
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers,
        'output_path': path,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(df_mibi),
        'media': media,
    }
    print('Temperatures:', calibr_presetup['exp_temps'])
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    calibr_setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    names = make_param_names(exps, bact, media)
    param_true = true_parameter_vector(path, add_name, media, n_cl)
    assert len(param_true) == len(calibr_setup['param_bnds']) == len(names)
    print(f'Cost (calibration objective) at the true parameters: '
          f'{fm.pest.cost_withS(param_true, calibr_setup, None):.6g}')

    # 2) global optimization ---------------------------------------------
    fm.pest.optimization.output_file = path + 'optimization_history1.csv'  # DE history into the test folder
    fixed_glob = resolve_fixed_values(FIX_PARAMS, names) if FIX_IN_GLOBAL else {}
    if RUN_GLOBAL:
        glob_setup = calibr_setup
        if fixed_glob:
            print_parameter_table(names, param_true, calibr_setup['param_bnds'], fixed_glob)
            glob_setup = fm.pest.apply_fixed_values(calibr_setup, fixed_glob, names=names)
        de_starts = []
        if GLOBAL_METHOD in ('de', 'de+multistart'):
            param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_withS, glob_setup)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            de_starts = [param_glob]
        if GLOBAL_METHOD in ('multistart', 'de+multistart'):
            param_glob, cost_glob = fm.pest.multistart_optimization(
                fm.pest.cost_withS, glob_setup, n_starts=N_STARTS, extra_starts=de_starts, method=LOCAL_METHOD,
                maxiter=200 if LOCAL_METHOD == 'least_squares' else 1000, n_jobs=n_jobs,
                history_file=path + 'optimization_history_multistart.csv')
        elif GLOBAL_METHOD != 'de':
            raise ValueError("GLOBAL_METHOD must be 'de', 'multistart' or 'de+multistart'")
        print(f'{(time.time() - start) / 60.:.1f} min (global), cost = {cost_glob:.6g}')
        s_x_glob = np.array(param_glob)[-n_cl*n_media:].reshape((n_media, n_cl))
        fm.output.json_dump({'param_ode': param_glob.astype(list), 's_x': s_x_glob, 'T_x': T_x, 'cost': cost_glob,
                             'fixed_params': fm.pest.fixed_values_record(fixed_glob, names)},
                            f'Result_calibration{add_name}_global.json', dir=path)
    else:
        param_glob = np.array(fm.output.read_from_json(f'Result_calibration{add_name}_global.json', dir=path)['param_ode'])
    print_comparison(names, param_glob, param_true, calibr_setup['param_bnds'], 'Global optimization vs. truth:')

    # 3) local refinement ------------------------------------------------
    if RUN_LOCAL:
        fixed = resolve_fixed_values(FIX_PARAMS, names)
        print_parameter_table(names, param_glob, calibr_setup['param_bnds'], fixed)
        param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup, fixed_values=fixed, maxiter=500 if LOCAL_METHOD == 'least_squares' else 1000,n_restarts=3, jitter_frac=0.1, n_jobs=n_jobs, method=LOCAL_METHOD, history_file=path + 'optimization_history_local.csv')
        print(f'{(time.time() - start) / 60.:.1f} min (global + local), cost = {cost_opt:.6g}')
        s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
        fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                             'fixed_params': fm.pest.fixed_values_record(fixed, names)},
                            f'Result_calibration{add_name}.json', dir=path)
    else:
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'])
    fixed = fixed_values_from_result(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path))
    print_comparison(names, param_opt, param_true, calibr_setup['param_bnds'], 'Local refinement vs. truth:')

    # 4) profile likelihood ----------------------------------------------
    if RUN_PROFILE:
        out_path = path + 'profile_likelihood/'
        os.makedirs(out_path, exist_ok=True)
        pl_setup = dict(calibr_setup)
        if fixed:  # parameters fixed in the refinement stay fixed (not profiled, not re-fitted)
            pl_setup, param_opt = fm.pest.apply_fixed_values(pl_setup, fixed, param=param_opt, names=names)
        pl_setup['aggregation_func'] = {
            'neg2logL': pl.make_gaussian_neg2loglik(pl_setup['data_array']),
            'geomean': pl.make_geomean_neg2loglik(pl_setup['data_array']),
            'calibration': fm.pest.cost_sum_and_geometric_mean}[OBJECTIVE]
        print(f'Profile objective ({OBJECTIVE}) at the truth: {fm.pest.cost_withS(param_true, pl_setup, None):.6g}, '
              f'at the estimate: {fm.pest.cost_withS(param_opt, pl_setup, None):.6g}')

        n_x0 = n_cl * len(exps)
        n_ode = 4 * n_cl + 2 + n_cl * n_cl
        idx_x0 = list(range(n_x0))
        idx_ode = list(range(n_x0, n_x0 + n_ode))
        idx_S = list(range(n_x0 + n_ode, len(param_opt)))
        profile_indices = {'x0': idx_x0, 'ode': idx_ode, 'ode+S': idx_ode + idx_S, 'all': None}[PROFILE]
        if FIX_X0 and PROFILE in ('x0', 'all'):
            raise ValueError('PROFILE=' + repr(PROFILE) + ' profiles x0, so FIX_X0 must be False')
        if FIX_S not in (False, 'estimate', 'true'):
            raise ValueError("FIX_S must be False, 'estimate' or 'true'")
        fixed_indices = (idx_x0 if FIX_X0 else []) + (idx_S if FIX_S else [])
        if FIX_S == 'true':  # S at the true values, the other parameters are re-fitted (refine=True)
            param_opt = np.array(param_opt, dtype=float)
            param_opt[idx_S] = param_true[idx_S]
        if FIX_S:
            print(f'S held fixed at the {FIX_S} values: {np.round(np.asarray(param_opt)[idx_S], 4)}')
        if profile_indices is None:
            profile_indices = list(range(len(param_opt)))
        profile_indices = [i for i in profile_indices if i not in fixed_indices]  # fixed ones can't be profiled
        tag = (f"{add_name}_{PROFILE.replace('+', '')}_{OBJECTIVE}" + ('_fixx0' if FIX_X0 else '')
               + {False: '', 'estimate': '_fixS', 'true': '_fixStrue'}[FIX_S])
        out_csv = out_path + f'profile_likelihood_results{tag}.csv'

        profile_options = ({'method': 'least_squares', 'maxiter': 200} if LOCAL_METHOD == 'least_squares'
                           else {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if EXTEND is not None:
            print('Parameter indices:')
            for i in profile_indices:
                print(f'  {i:3d}  {names[i]:<28} estimate = {param_opt[i]:.6g}, bounds {calibr_setup["param_bnds"][i]}')
            pl.extend_profile_likelihood(
                fm.pest.cost_withS, pl_setup, out_csv, EXTEND,
                n_jobs=n_jobs, n_restarts=1, jitter_frac=0.05,
                fixed_indices=fixed_indices,
                stop_factor=3,
                minimize_options=profile_options,
                plot_path=out_path + f'profile_likelihood{tag}.png',
                true_params=param_true if SHOW_TRUE_VALUE else None,
                auto_n_points=EXTEND_N_POINTS,
                verbose=1,
            )
        else:
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
                refine=False or FIX_S == 'true',  # re-fit first (always when FIX_S='true', which moves the optimum)
                stop_factor=3,                 # stop each side once Delta > 1.5 x threshold
                minimize_options=profile_options,
                true_params=param_true if SHOW_TRUE_VALUE else None,
                verbose=1,
                restart_on_better=False,
                resume=True,
            )
        pl.plot_profile_likelihood_individual(out_csv, out_dir=out_path + 'individual', file_ext='pdf',
                                              true_params=param_true if SHOW_TRUE_VALUE else None)
        print(f'{(time.time() - start) / 60.:.1f} min in total')

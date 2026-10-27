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
import paper_paths as pp
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import (make_param_names, exp_temps_from_columns, resolve_fixed_values,
                                            fixed_values_from_result, print_parameter_table,
                                            copy_result_to_all_profiles)

import numpy as np


# ----------------------------------------------------------------------
# True model (2 species; values of species 00 and 01 of the 4-species in-silico model)
# ----------------------------------------------------------------------
N_CL = 2
# true ODE parameters of species 00 and 01 [lambda_1, lambda_exp, alpha_0, alpha_exp, N_1, N_exp, k_ij]
# from model_paper/setup/True_parameters_paper.json (created by main_true_parameters.py)
PARAM_ODE_TRUE = fm.data.read_true_parameters(N_CL)['param_ode']
GEN_MEDIA = ['sel1', 'gen1']  # media order of fm.data.model_2media_expgensel (S rows in the true file)

# Experimental designs of the in-silico data (each design has its own output folder, so that the designs can
# be compared; 'base' = model_paper/out/test_2sp_<noise>noise/). Replicates = repeated temperatures, each
# experiment with its own initial values (max. 15 experiments). 'days': measurement days (whole days),
# default 0, 1, ..., 17. Also used by test_2sp_sepS.py.
DESIGNS = {
    'base':        {'temps': [2., 6., 10.]},                                         # 3 experiments
    'rep3':        {'temps': [2., 2., 2., 6., 6., 6., 10., 10., 10.]},                # 3 replicates each
    'temps6':      {'temps': [2., 4., 6., 8., 10., 14.]},                            # more temperatures
    'temps6_rep2': {'temps': [2., 2., 4., 4., 6., 6., 8., 8., 10., 10., 14., 14.]},  # both
    'long':        {'temps': [2., 6., 10.], 'days': list(range(0, 31))},             # 30 days
}


def data_folder(design, relnoise, normalize_true_s=False):
    """Output / data folder of test_2sp.py for a design and noise level (see paper_paths.test_2sp_path)."""
    return pp.test_2sp_path(design, relnoise, normalize_true_s)


def true_parameter_vector(path, add_name, media_sorted, n_cl):
    """True parameters in the layout of the calibration vector [x0 | ODE | S (media sorted)]."""
    tr = fm.output.read_from_json(f'Result_real{add_name}.json', dir=path)
    s_true = np.array(tr['s_x'], dtype=float).reshape(len(GEN_MEDIA), n_cl)
    s_true = np.array([s_true[GEN_MEDIA.index(m)] for m in media_sorted])  # calibration uses sorted media
    return np.concatenate([np.ravel(tr['param_ode']), s_true.ravel()])


def normalize_truth_max_s(S_matrix_setup, x10, param_ode, n_cl):
    """
    Rescale the true parameters so that the largest media coefficient is 1 (as S_MAX_PENALTY assumes), with
    (almost) unchanged observables: S / m, counts * m, i.e. x0 + log10(m), N_1 + log10(m), k_ij / m, where
    m = max(S) of the media used (sel1, gen1). Exact up to the one initially active cell per species.
    Returns (S_matrix_setup, x10, param_ode, m) with copies of the changed values.
    """
    sel = np.array(S_matrix_setup['s_selective'][0][:n_cl], dtype=float)
    gen = np.array(S_matrix_setup['s_general'][0][:n_cl], dtype=float)
    m = float(max(sel.max(), gen.max()))
    S2 = dict(S_matrix_setup)
    S2['s_selective'] = [list(r) for r in S_matrix_setup['s_selective']]
    S2['s_general'] = [list(r) for r in S_matrix_setup['s_general']]
    S2['s_selective'][0][:n_cl] = list(sel / m)
    S2['s_general'][0][:n_cl] = list(gen / m)
    p = np.array(param_ode, dtype=float)
    p[4 * n_cl] += np.log10(m)          # N_1
    p[4 * n_cl + 2:] = p[4 * n_cl + 2:] / m  # k_ij
    return S2, np.asarray(x10, dtype=float) + np.log10(m), p, m


def rescale_truth_max_s(param, n_cl, n_exps, n_media):
    """
    The full true parameter vector [x0 | ODE | S] rescaled to max(S) = 1, for comparing estimates obtained with
    S_MAX_PENALTY (the data determine S only up to a common factor; the penalty fixes max(S) = 1):
    S / m, x0 and N_1 + log10(m), k_ij / m with m = max(S). Not used for generating data.
    """
    p = np.array(param, dtype=float)
    n_x0, n = n_cl * n_exps, n_cl
    m = float(np.max(p[-n_cl * n_media:]))
    p[:n_x0] += np.log10(m)
    p[n_x0 + 4 * n] += np.log10(m)
    p[n_x0 + 4 * n + 2:n_x0 + 4 * n + 2 + n * n] /= m
    p[-n_cl * n_media:] /= m
    return p


def print_comparison(names, estimate, true, bnds, title, true_rescaled=None):
    """Table estimate vs. truth; true_rescaled: additional column, e.g. the truth rescaled to max(S) = 1."""
    print(f'\n{title}')
    extra = f'{"true (max S=1)":>16}' if true_rescaled is not None else ''
    print(f'{"parameter":<22}{"true":>10}{extra}{"estimate":>12}{"bounds":>18}')
    tr = true_rescaled if true_rescaled is not None else [None] * len(names)
    for n, t, r, e, b in zip(names, true, tr, estimate, bnds):
        if b[1] > b[0]:
            flag = '  <- at bound' if np.isclose(e, b[0]) or np.isclose(e, b[1]) else ''
            col = f'{r:>16.4g}' if r is not None else ''
            print(f'{n:<22}{t:>10.4g}{col}{e:>12.4g}   ({b[0]:g}, {b[1]:g}){flag}')


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    # Experimental design of the in-silico data (each design has its own output folder, so that the
    # designs can be compared; 'base' = model_paper/out/test_2sp/ as before). Replicates = repeated
    # temperatures, each experiment with its own initial values (max. 15 experiments). 'days': measurement
    # days (whole days), default 0, 1, ..., 17. After switching to a new design: RUN_DATA = True.

    DESIGN = 'rep3'
    temps = DESIGNS[DESIGN]['temps']
    days_meas = DESIGNS[DESIGN].get('days')  # None: default 0..17
    n_cl = N_CL
    n_media = 2
    relnoise = 0.1
    ntr = 1                        # keep 1: replicates via repeated temperatures (see DESIGNS)
    add_name = f'_{n_cl}dim_{n_media}media_test'
    workers = 22#-1                   # differential evolution: -1 = all cores
    n_jobs = 22#os.cpu_count()-2 or 1   # local multistart / profile likelihood processes

    RUN_DATA = True               # generate the in-silico data
    RUN_GLOBAL = True             # 2) differential evolution
    RUN_LOCAL = True              # 3) local refinement
    RUN_PROFILE = True             # 4) profile likelihood

    # optimizer settings
    # global step: 'de'            differential evolution (as before)
    #              'multistart'    N_STARTS local fits from Latin hypercube points (recommended for many species)
    #              'de+multistart' differential evolution, then the multistart incl. the DE result
    GLOBAL_METHOD = 'de'
    N_STARTS = 1
    # local step and profile fits: 'L-BFGS-B' (as before) or 'least_squares' (trust-region least squares on
    # the residuals, parameters scaled, much faster convergence for ill-conditioned problems)
    LOCAL_METHOD = 'least_squares'
    # derivatives of the fits: 'numpy' (finite differences, as before) or 'jax' (exact derivatives by automatic
    # differentiation, much faster and a more accurate ODE solve; needs: pip install jax diffrax). Used for the
    # local step, multistart and the profiles (not for differential evolution). Don't mix backends within one
    # profile run: the profile tools check the cost at the stored optimum and stop if it differs.
    BACKEND = 'jax'
    # local step: parameters held at a fixed value, only the others are refined (e.g. non-identifiable ones
    # set to 0). Keys: index or name (see the printed parameter table), e.g. {17: 0., 18: 0.} or
    # {'k_00,01': 0.}; the value may lie outside the bounds. Stored as 'fixed_params' in
    # Result_calibration*.json and kept fixed in the profile likelihood (not profiled, not re-fitted).
    #FIX_PARAMS = {'k_00,01': 0.,
    #              'alpha_exp,00': 0.,
    #              'lambda_exp,00': 0.}
    FIX_PARAMS = {}#'lambda_exp,01': 0.}
    # True: FIX_PARAMS also in the global step (new differential evolution / multistart search with these
    # parameters fixed) instead of only refining from the earlier global result
    FIX_IN_GLOBAL = True
    # Penalty so that the largest media coefficient S is 1 (fixes the x0/S scale trade-off): weight w, the cost
    # rises by the factor exp(w * (max S - 1)^2), e.g. w = 100: max S = 0.9 -> +172 %, 0.97 -> +9 %. None: off.
    # Stored in the result JSON; the profile likelihood uses the same penalty (needs new profiles).
    S_MAX_PENALTY = 100
    # False (default): data from the actual truth; with S_MAX_PENALTY the comparison tables show an extra column
    #   'true (max S=1)': the truth rescaled to max(S) = 1, which is what the penalized fit estimates.
    # True: generate the data from the rescaled truth (S / m, x0 and N_1 + log10(m), k / m; same observables),
    #   output folder suffix '_normS' (e.g. to test that a noise-free fit with the penalty recovers the truth).
    NORMALIZE_TRUE_S = False
    # Noise model of the residuals (all optimizations and the profiles):
    #   'absolute': measured - model (MiBi divided by its maximum), as before
    #   'relative': multiplicative noise (measured = model * (1 + eps)): log residuals for MiBi, centred
    #               log-ratios within each composition for MALDI / NGS -> constant variance, valid chi2 threshold
    NOISE_MODEL = 'relative'

    # profile likelihood settings
    PROFILE = 'all'              # 'x0', 'ode', 'ode+S' or 'all' (= x0 + ode + S)
    FIX_X0 = False                 # True: x0 held at the estimate (faster, profiles conditional on x0)
    # S (media matrix): False = re-fitted at every profile point (default)
    #                   'estimate' = held at the estimated values, 'true' = held at the true values (in-silico)
    FIX_S = False        # S re-fitted at every profile point (as before)
    #FIX_S = 'estimate'   # S held at the estimated values
    #FIX_S = 'true'       # S held at the true values (in-silico data only)  
    # Objective of the profiles and (CALIBRATION_OBJECTIVE = OBJECTIVE) of all optimizations:
    # 'neg2logL'   : -2 log L, data types weighted by their number of points
    # 'geomean'    : -2 log L form of cost_sum_and_geometric_mean (same optimum, equal weight per data type)
    # 'calibration': cost_sum_and_geometric_mean itself (profile scale linearized: G_opt / N)
    # 'known_noise': -2 log L with a KNOWN noise level KNOWN_SIGMA (sum SSR_k / sigma_k^2), e.g. for noise-free
    #                data (relnoise = 0): the profiles of an experiment with that noise level. The objectives
    #                above estimate the noise from the residuals and diverge for noise-free data.
    OBJECTIVE = 'neg2logL'
    # noise level for OBJECTIVE = 'known_noise': one value or [NGS, MALDI, MiBi], in the units of the residuals
    # (NOISE_MODEL = 'relative': standard deviation of the log residuals, 0.1 ~ 10 % relative noise)
    KNOWN_SIGMA = 0.1
    # objective of differential evolution / multistart / local refinement; OBJECTIVE = the same objective
    # everywhere (the optimum of the calibration is then directly the optimum of the profiles).
    # 'calibration' = cost_sum_and_geometric_mean as before. Stored as 'objective' in the result JSON files;
    # an existing result with a different objective is kept as Result_calibration<...>_<objective>.json.
    CALIBRATION_OBJECTIVE = OBJECTIVE
    SHOW_TRUE_VALUE = True         # False: profile plots without the green "true value" line
    # Add points to the existing profiles (same PROFILE / FIX_* / OBJECTIVE -> same CSV) instead of a new run:
    #   None   : normal profile run
    #   'auto' : every profile side that does not reach the threshold (and does not end at the bound)
    #            gets EXTEND_N_POINTS new points up to the bound; a side stops once Delta > 3 x threshold
    #   dict   : {index or name: spec} (name as in the CSV columns, without $ \ { }, e.g. 'x_0,00^V01')
    #            with spec = {'upper': value, 'lower': value, 'n_points': n} (new points from
    #            the current end of the profile to value) or a list of values (anywhere, also to fill gaps);
    #            add 'auto': True to also extend all other parameters automatically.
    #            spec = {'ci': n, 'margin': m} (or just 'ci' = 10 points): n evenly spaced points over
    #            the current CI, widened by m (default 0.3) x CI half-width on each side (denser
    #            profile inside and slightly outside the CI; combinable with the other keys).
    #            Indices are printed as "Parameter indices" when EXTEND is set.
    #   'ci'   : automatic CI densification: every profile with fewer than min_points points inside its CI
    #            gets denser (n_points values over the CI + margin), repeated with twice as many values
    #            for the profiles still below min_points, at most max_rounds rounds. As a dict key, e.g.
    #            {'auto': True, 'ci': {'n_points': 10, 'margin': 0.3, 'min_points': 5, 'max_rounds': 3}}
    #            ('ci': True = these defaults); runs after the other entries of the dict.
    #   e.g. EXTEND = {2: {'upper': 5.0, 'n_points': 6}, 12: [0.01, 0.03, 0.1, 0.3, 0.6], 'auto': True}
    #        EXTEND = {3: 'ci', 'k_01,00': {'ci': 15, 'margin': 0.5}}
    # EXTEND_MODE selects between the two prepared options below:
    #   None     : no extension (normal profile run)
    #   'auto'   : EXTEND_AUTO, profiles chosen automatically (open sides + too few points in the CI)
    #   'manual' : EXTEND_MANUAL, profiles and specs chosen by hand
    EXTEND_MODE = 'auto'
    EXTEND_AUTO = {'auto': True, 'ci': {'n_points': 10, 'margin': 0.3, 'min_points': 5, 'max_rounds': 3}}
    # rep3 (profile_likelihood_2dim_2media_test_all_neg2logL), after the first extension ({'ci': 5} + 'auto'):
    # all CIs closed (S_sel1,01 ends at its bound 1), but these profiles still had only 3-7 points in the
    # CI region (CI + 30 % margin)
    EXTEND_MANUAL = {name: {'ci': 7, 'margin': 0.1} for name in [
        'x_0,00^V01', 'x_0,01^V01', 'x_0,00^V02', 'x_0,01^V02', 'x_0,01^V03', 'x_0,00^V04', 'x_0,01^V04',
        'x_0,00^V05', 'x_0,01^V05', 'x_0,00^V06', 'x_0,01^V06', 'x_0,00^V07', 'x_0,01^V08', 'x_0,01^V09',
        'lambda_1,00', 'alpha_0,00', 'alpha_0,01', 'alpha_exp,00', 'alpha_exp,01', 'N_1',
        'S_gen1,01']}
    EXTEND = {None: None, 'auto': EXTEND_AUTO, 'manual': EXTEND_MANUAL}[EXTEND_MODE]
    EXTEND_N_POINTS = 6
    # True: use the current Result_calibration*.json (e.g. a new, better local minimization) as the optimum of
    # the existing profiles (same PROFILE / FIX_* / OBJECTIVE -> same CSV) instead of computing profiles:
    # optimum rows, cost_opt, summary JSON (CIs) and plots are updated, the profile points are kept.
    UPDATE_OPTIMUM = False
    # Refit the points of the existing profiles from their neighbours (removes jumps where points ended in a
    # worse local minimum; no new values): False, True (all profiles) or a list of indices / names, e.g.
    # ['k_01,00']. Runs after the profile run / EXTEND / UPDATE_OPTIMUM. Backup: <csv>_before_polish.csv
    POLISH_PROFILES = True
    # ================================================================

    path = data_folder(DESIGN, relnoise, NORMALIZE_TRUE_S)
    os.makedirs(path, exist_ok=True)
    start = time.time()

    # 1) in-silico data -------------------------------------------------
    S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=fm.data.SETUP_DIR)
    x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=fm.data.SETUP_DIR)['x0'])[:len(temps), :n_cl]
    if len(x10) < len(temps):
        raise ValueError(f'design {DESIGN!r} has {len(temps)} experiments, the setup has initial values for {len(x10)}')
    print(f'Design {DESIGN!r}: temperatures {temps}, days {days_meas if days_meas is not None else "0..17"}, output {path}')
    param_ode_gen = PARAM_ODE_TRUE.copy()
    if NORMALIZE_TRUE_S:
        S_matrix_setup, x10, param_ode_gen, m_s = normalize_truth_max_s(S_matrix_setup, x10, param_ode_gen, n_cl)
        print(f'True parameters rescaled to max(S) = 1 (factor {m_s:g}): x0, N_1 + {np.log10(m_s):.4f}, S and k / {m_s:g}')
    if RUN_DATA or not os.path.exists(path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup, param_ode=param_ode_gen,
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=path, add_name=add_name,
            t_meas=days_meas)
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
        #'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(df_mibi),
        'media': media,
    }
    print('Temperatures:', calibr_presetup['exp_temps'])
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    calibr_setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    calibr_setup['s_max_penalty'] = S_MAX_PENALTY
    calibr_setup['noise_model'] = NOISE_MODEL

    def make_objective(name):
        """Aggregation function of an objective name (see OBJECTIVE)."""
        return {'neg2logL': pl.make_gaussian_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'geomean': pl.make_geomean_neg2loglik(calibr_setup['data_array'], NOISE_MODEL),
                'calibration': fm.pest.cost_sum_and_geometric_mean,
                'known_noise': pl.make_known_noise_neg2loglik(calibr_setup['data_array'], KNOWN_SIGMA, NOISE_MODEL)}[name]

    def save_result(fname, data):
        """Write a result JSON with its objective and noise model; keep an existing file that has a different
        objective or noise model (as <file>_<objective>[_relative].json)."""
        f = os.path.join(path, fname)
        if os.path.exists(f):
            old = fm.output.read_from_json(fname, dir=path)
            old_nm = old.get('noise_model', 'absolute')
            old_obj = old.get('objective', 'calibration') + ('' if old_nm == 'absolute' else f'_{old_nm}')
            if old_obj != CALIBRATION_OBJECTIVE + ('' if NOISE_MODEL == 'absolute' else f'_{NOISE_MODEL}'):
                bak = f.replace('.json', f'_{old_obj}.json')
                if not os.path.exists(bak):
                    import shutil
                    shutil.copy(f, bak)
                    print(f'Kept the previous result (objective {old_obj}) as {bak}')
        fm.output.json_dump(dict(data, objective=CALIBRATION_OBJECTIVE, noise_model=NOISE_MODEL, backend=BACKEND),
                            fname, dir=path)

    calibr_setup['aggregation_func'] = make_objective(CALIBRATION_OBJECTIVE)
    print(f'Objective of the optimizations: {CALIBRATION_OBJECTIVE}, noise model: {NOISE_MODEL}')
    names = make_param_names(exps, bact, media)
    param_true = true_parameter_vector(path, add_name, media, n_cl)
    assert len(param_true) == len(calibr_setup['param_bnds']) == len(names)
    print(f'Cost ({CALIBRATION_OBJECTIVE}) at the true parameters: '
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
            param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_withS, glob_setup, maxiter=500)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            de_starts = [param_glob]
        if GLOBAL_METHOD in ('multistart', 'de+multistart'):
            param_glob, cost_glob = fm.pest.multistart_optimization(
                fm.pest.cost_withS, glob_setup, n_starts=N_STARTS, extra_starts=de_starts, method=LOCAL_METHOD,
                backend=BACKEND,
                maxiter=200 if LOCAL_METHOD == 'least_squares' else 500, n_jobs=n_jobs,
                history_file=path + 'optimization_history_multistart.csv')
        elif GLOBAL_METHOD != 'de':
            raise ValueError("GLOBAL_METHOD must be 'de', 'multistart' or 'de+multistart'")
        print(f'{(time.time() - start) / 60.:.1f} min (global), cost = {cost_glob:.6g}')
        s_x_glob = np.array(param_glob)[-n_cl*n_media:].reshape((n_media, n_cl))
        save_result(f'Result_calibration{add_name}_global.json',
                    {'param_ode': param_glob.astype(list), 's_x': s_x_glob, 'T_x': T_x, 'cost': cost_glob,
                     'fixed_params': fm.pest.fixed_values_record(fixed_glob, names), 's_max_penalty': S_MAX_PENALTY})
    else:
        param_glob = np.array(fm.output.read_from_json(f'Result_calibration{add_name}_global.json', dir=path)['param_ode'])
    # with S_MAX_PENALTY the estimates correspond to the truth rescaled to max(S) = 1 (extra column)
    param_true_rs = (rescale_truth_max_s(param_true, n_cl, len(exps), n_media)
                     if S_MAX_PENALTY and not NORMALIZE_TRUE_S else None)
    print_comparison(names, param_glob, param_true, calibr_setup['param_bnds'], 'Global optimization vs. truth:',
                     true_rescaled=param_true_rs)

    # 3) local refinement ------------------------------------------------
    if RUN_LOCAL:
        fixed = resolve_fixed_values(FIX_PARAMS, names)
        print_parameter_table(names, param_glob, calibr_setup['param_bnds'], fixed)
        param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_withS, param_glob, calibr_setup, fixed_values=fixed, maxiter=400 if LOCAL_METHOD == 'least_squares' else 1000,n_restarts=5, jitter_frac=0.1, n_jobs=n_jobs, method=LOCAL_METHOD, backend=BACKEND, history_file=path + 'optimization_history_local.csv')
        print(f'{(time.time() - start) / 60.:.1f} min (global + local), cost = {cost_opt:.6g}')
        s_x = np.array(param_opt)[-n_cl*n_media:].reshape((n_media, n_cl))
        save_result(f'Result_calibration{add_name}.json',
                    {'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt,
                     'fixed_params': fm.pest.fixed_values_record(fixed, names), 's_max_penalty': S_MAX_PENALTY})
        copy_result_to_all_profiles(path, add_name)  # new optimum into existing profile JSON files
    else:
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'])
    result_local = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)
    fixed = fixed_values_from_result(result_local)
    result_objective = result_local.get('objective', 'calibration')
    result_noise_model = result_local.get('noise_model', 'absolute')
    result_backend = result_local.get('backend', 'numpy')
    print_comparison(names, param_opt, param_true, calibr_setup['param_bnds'], 'Local refinement vs. truth:',
                     true_rescaled=param_true_rs)

    # 4) profile likelihood ----------------------------------------------
    if RUN_PROFILE:
        out_path = path + 'profile_likelihood/'
        os.makedirs(out_path, exist_ok=True)
        pl_setup = dict(calibr_setup)
        if fixed:  # parameters fixed in the refinement stay fixed (not profiled, not re-fitted)
            pl_setup, param_opt = fm.pest.apply_fixed_values(pl_setup, fixed, param=param_opt, names=names)
        pl_setup['aggregation_func'] = make_objective(OBJECTIVE)
        # 'geomean' and 'calibration' have the same optimum; otherwise the estimate is re-fitted first
        same_optimum = (({result_objective, OBJECTIVE} <= {'geomean', 'calibration'} or result_objective == OBJECTIVE)
                        and result_noise_model == NOISE_MODEL and result_backend == BACKEND)
        if not same_optimum:
            print(f'Result_calibration was optimized with {result_objective!r} / noise model {result_noise_model!r}, '
                  f'the profiles use {OBJECTIVE!r} / {NOISE_MODEL!r}: the estimate is re-fitted first (refine)')
        print(f'Profile objective ({OBJECTIVE}) at the truth: {fm.pest.cost_withS(param_true, pl_setup, None):.6g}, '
              f'at the estimate: {fm.pest.cost_withS(param_opt, pl_setup, None):.6g}')
        # breakdown per data type: sum of squared residuals and number of data points (degrees of freedom)
        n_dof = fm.pest.count_data_points_noise(pl_setup['data_array'], NOISE_MODEL)
        for lab, p_ in (('truth', param_true), ('estimate', param_opt)):
            groups, _ = fm.pest.residuals_withS(p_, pl_setup)
            print(f'  {lab:8s}: ' + ', '.join(f'{k} SSR = {np.sum(r**2):.4g} (n = {n})'
                                              for k, r, n in zip(('NGS', 'MALDI', 'MiBi'), groups, n_dof))
                  + f', max S = {np.max(np.asarray(p_)[-n_cl*n_media:]):.3g}')

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
        profile_options['backend'] = BACKEND
        if UPDATE_OPTIMUM:
            pl.update_profile_optimum(fm.pest.cost_withS, pl_setup, out_csv, param_opt, backend=BACKEND,
                                      true_params=param_true if SHOW_TRUE_VALUE else None,
                                      plot_path=out_path + f'profile_likelihood{tag}.png')
        elif EXTEND is not None:
            print('Parameter indices:')
            for i in profile_indices:
                print(f'  {i:3d}  {names[i]:<28} estimate = {param_opt[i]:.6g}, bounds {calibr_setup["param_bnds"][i]}')
            pl.extend_profile_likelihood(
                fm.pest.cost_withS, pl_setup, out_csv, EXTEND,
                n_jobs=n_jobs, n_restarts=3, jitter_frac=0.05,
                fixed_indices=fixed_indices,
                stop_factor=1.5,
                minimize_options=profile_options,
                plot_path=out_path + f'profile_likelihood{tag}.png',
                true_params=param_true if SHOW_TRUE_VALUE else None,
                auto_n_points=EXTEND_N_POINTS,
                verbose=1,
            )
        else:
            df, cis = pl.run_profile_likelihood_all(
                fm.pest.cost_withS, param_opt, pl_setup,
                span=0.3, n_points=35, method='local',
                n_jobs=n_jobs,
                out_csv=out_csv,
                plot_path=out_path + f'profile_likelihood{tag}.png',
                param_names=names,
                n_restarts=3, jitter_frac=0.05,
                profile_indices=profile_indices,
                fixed_indices=fixed_indices,
                refine=FIX_S == 'true' or not same_optimum,  # re-fit first if the optimum differs (FIX_S='true', objective)
                stop_factor=1.5,                 # stop each side once Delta > 1.5 x threshold
                minimize_options=profile_options,
                true_params=param_true if SHOW_TRUE_VALUE else None,
                verbose=1,
                restart_on_better=False,
                resume=False,
            )
        if POLISH_PROFILES:
            polish_idx = (None if POLISH_PROFILES is True
                          else sorted(resolve_fixed_values({k: 0. for k in POLISH_PROFILES}, names)))
            pl.polish_profile_likelihood(fm.pest.cost_withS, pl_setup, out_csv, profile_indices=polish_idx,
                                         n_jobs=n_jobs, fixed_indices=fixed_indices,
                                         minimize_options=profile_options,
                                         true_params=param_true if SHOW_TRUE_VALUE else None,
                                         plot_path=out_path + f'profile_likelihood{tag}.png')
        pl.plot_profile_likelihood_individual(out_csv, out_dir=out_path + 'individual', file_ext='pdf',
                                              true_params=param_true if SHOW_TRUE_VALUE else None)
        print(f'{(time.time() - start) / 60.:.1f} min in total')

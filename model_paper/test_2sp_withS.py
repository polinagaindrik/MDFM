"""
Test case: 2-species model, all parameters estimated together with the S-step cost terms
========================================================================================

Pipeline of test_2sp.py (x0, ODE parameters and S estimated together), with the cost of the separate S
estimation (test_2sp_sepS.py, media_matrix.S_squared_difference) added to the objective:

  data groups:  NGS, MALDI, MiBi                         (cost_withS, as in test_2sp.py)
  S groups:     NGS/MALDI relation  (Scost_ngsterm)       (S_squared_difference, residuals divided by |S|)
                MiBi/MALDI media relation (Scost_mediarelation)
  (+ the S_MAX_PENALTY group, as in test_2sp.py)

Parameter vector [x0 | ODE | S (media sorted)], as in test_2sp.py. The media relation term ties the ratio of
the S rows (gen1 vs. sel1) to MiBi x MALDI on both media; the S groups depend on S only, not on x0 / ODE.
The objectives are -2 log L forms with one group per residual type (each with its own variance), so the
S terms enter like two further data types; 'calibration' (cost_sum_and_geometric_mean) is not offered.

  1) in-silico data: as test_2sp.py; DESIGN and relnoise select the same data folder as there (shared),
     results go to <data folder>_withS/
  2) global optimization (differential evolution)       -> Result_calibration<add_name>_global.json
  3) local refinement (least_squares / L-BFGS-B)         -> Result_calibration<add_name>.json
  4) profile likelihood                                  -> profile_likelihood/...

Derivatives by finite differences only (BACKEND = 'numpy'): the JAX backend has no S terms.
Each stage can be switched off (RUN_*) and is then read from its saved result. Run from the repository root:
    python model_paper/test_2sp_withS.py
"""
import os
import sys
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from fusion_model.parameter_estimation import least_squares_fit as lsf
from profile_likelihood_model_paper import (make_param_names, exp_temps_from_columns, resolve_fixed_values,
                                            fixed_values_from_result, print_parameter_table,
                                            copy_result_to_all_profiles)
from test_2sp import (N_CL, PARAM_ODE_TRUE, DESIGNS, data_folder, true_parameter_vector,
                      normalize_truth_max_s, rescale_truth_max_s, print_comparison)

import numpy as np
import pandas as pd

S_GROUPS = ('S_ngs', 'S_media')  # names of the S residual groups (order as in S_squared_difference)


# ----------------------------------------------------------------------
# Cost with the S-step terms, in the cost_func(param, calibr_setup, jac_spasity) form
# ----------------------------------------------------------------------

def _S_term_raw(s_x, calibr_setup):
    """Signed S-step residuals [ngs term, media relation term] (all entries, NaN where data are missing) for
    S = s_x: the square roots of media_matrix.S_squared_difference(S)."""
    mm = fm.pest.media_matrix
    days, (obs_mibi, obs_maldi, obs_ngs) = calibr_setup['data_array']
    mibi_max = np.nanmax(obs_mibi, axis=(0, 1))
    norm = np.sqrt(np.nansum(s_x**2))
    r_ngs, r_media = [], []
    for i in range(len(obs_mibi)):
        rhs1, lhs1 = mm.Scost_ngsterm(days, obs_maldi[i], obs_ngs[i], s_x, calibr_setup['T_x'])
        rhs2, lhs2 = mm.Scost_mediarelation(days, obs_mibi[i]/mibi_max, obs_maldi[i], s_x)
        r_ngs.append(np.ravel(rhs1 - lhs1))
        r_media.append(np.ravel(rhs2 - lhs2))
    return [np.concatenate(r_ngs)/norm, np.concatenate(r_media)/norm]


def S_term_masks(calibr_setup):
    """Entries of the S residual groups that are used (finite; depends only on the data)."""
    s_ones = np.ones((calibr_setup['n_media'], calibr_setup['n_cl']))
    return [np.isfinite(r) for r in _S_term_raw(s_ones, calibr_setup)]


def residuals_withS_Sterms(param, calibr_setup, jac_spasity=None):
    """
    Signed residuals of cost_withS_Sterms for param = [x0 | ODE | S]: the groups of fm.pest.residuals_withS
    [NGS, MALDI, MiBi] followed by the S groups [S_ngs, S_media] (owner -1: independent of x0), then the
    s_max penalty group if calibr_setup['s_max_penalty'] is set.
    calibr_setup['s_term_masks'] = S_term_masks(calibr_setup).
    """
    param = np.asarray(param, dtype=float)
    res, owner = fm.pest.residuals_withS(param, calibr_setup, jac_spasity)
    n_cl, n_media = calibr_setup['n_cl'], calibr_setup['n_media']
    s_x = param[-n_cl*n_media:].reshape((n_media, n_cl))
    r_S = [r[m] for r, m in zip(_S_term_raw(s_x, calibr_setup), calibr_setup['s_term_masks'])]
    n_data = 3
    res = res[:n_data] + r_S + res[n_data:]
    owner = owner[:n_data] + [np.full(len(r), -1) for r in r_S] + owner[n_data:]
    return res, owner


def cost_withS_Sterms(param, calibr_setup, jac_spasity=None):
    """cost_withS plus the S-step terms: calibr_setup['aggregation_func'] of the squared residual groups of
    residuals_withS_Sterms (incl. the penalty group, see SMaxPenaltyNeg2LogLik)."""
    res, _ = residuals_withS_Sterms(param, calibr_setup, jac_spasity)
    if not all(np.all(np.isfinite(r)) for r in res):
        return pl.PENALTY
    return calibr_setup['aggregation_func']([r**2 for r in res])


lsf.RESIDUAL_FUNCS[cost_withS_Sterms] = residuals_withS_Sterms  # for method='least_squares' (fits, profiles)


class SMaxPenaltyNeg2LogLik(pl.GaussianNeg2LogLik):
    """
    -2 log L objective `base` of all groups but the last, plus the s_max penalty from the last group
    ((max S - 1)^2), as cost_function.apply_s_max_penalty: base + N * w * (max S - 1)^2 (N = data points).
    A GaussianNeg2LogLik, so that the profile likelihood uses scale = 1.
    """
    def __init__(self, base, w):
        self.base = base
        self.w = float(w)
        self.n_data = list(base.n_data)

    def __call__(self, J_vect):
        return self.base(J_vect[:-1]) + sum(self.n_data) * self.w * float(np.sum(J_vect[-1]))


if __name__ == "__main__":
    # ================================================================
    # Settings (as in test_2sp.py)
    # ================================================================
    # in-silico data: the same design and noise level as in test_2sp.py -> same data folder (shared), the
    # results of this script go to <data folder>_withS/
    DESIGN = 'rep3'
    temps = DESIGNS[DESIGN]['temps']
    days_meas = DESIGNS[DESIGN].get('days')  # None: default 0..17
    n_cl = N_CL
    n_media = 2
    relnoise = 0.1
    ntr = 1                        # keep 1: replicates via repeated temperatures (see DESIGNS)
    add_name = f'_{n_cl}dim_{n_media}media_test'
    workers = 22                   # differential evolution: -1 = all cores
    n_jobs = 22                    # local multistart / profile likelihood processes

    RUN_DATA = False               # 1) generate the in-silico data (also done if no data in the data folder)
    RUN_GLOBAL = False              # 2) differential evolution
    RUN_LOCAL = True               # 3) local refinement
    RUN_PROFILE = True             # 4) profile likelihood

    # optimizer settings (see test_2sp.py)
    # 'multistart' (least_squares from Latin hypercube points, one start per process) takes minutes, differential
    # evolution (popsize 30 x 36 parameters, 0.07 s per cost evaluation) about half an hour or more
    GLOBAL_METHOD = 'multistart'   # 'de', 'multistart' or 'de+multistart'
    N_STARTS = 21                  # + the warm start: 22 starts = n_jobs, all in parallel
    # True: also start from the optimum of test_2sp.py (same data, without the S terms), if it exists
    WARM_START = True
    LOCAL_METHOD = 'least_squares' # 'L-BFGS-B' or 'least_squares'
    BACKEND = 'numpy'              # only 'numpy': the JAX backend does not contain the S terms
    FIX_PARAMS = {}                # parameters held at a fixed value (index or name), e.g. {'k_00,01': 0.}
    FIX_IN_GLOBAL = True
    S_MAX_PENALTY = 100            # penalty so that max(S) = 1 (None: off)
    NORMALIZE_TRUE_S = False       # data from the truth rescaled to max(S) = 1 (folder suffix '_normS')
    NOISE_MODEL = 'relative'       # 'absolute' or 'relative' (data groups only; the S groups stay as they are)

    # profile likelihood settings
    PROFILE = 'all'                # 'x0', 'ode', 'ode+S' or 'all' (= x0 + ode + S)
    FIX_X0 = False
    FIX_S = False                  # False, 'estimate' or 'true'
    # Objective of the profiles and (CALIBRATION_OBJECTIVE = OBJECTIVE) of all optimizations, each with one
    # group per residual type [NGS, MALDI, MiBi, S_ngs, S_media]:
    # 'neg2logL'   : -2 log L, groups weighted by their number of points
    # 'geomean'    : -2 log L with equal weight per group
    # 'known_noise': -2 log L with known noise levels KNOWN_SIGMA (data groups) and KNOWN_SIGMA_S (S groups)
    OBJECTIVE = 'geomean'            # equal weight per group (S terms: 2 of 5 groups)
    KNOWN_SIGMA = 0.1              # one value or [NGS, MALDI, MiBi]
    KNOWN_SIGMA_S = 0.1            # one value or [S_ngs, S_media], in the units of the S residuals
    CALIBRATION_OBJECTIVE = OBJECTIVE
    SHOW_TRUE_VALUE = True
    # Profile run in three steps (run_profile_likelihood_adaptive, as in test_2sp_sepS.py), grid plot after
    # each step (<plot>_1step.png, _2step, _3step):
    #   1. profiles on the initial grid (span, n_points of the run)
    #   2. sides that do not cross the threshold get EXTEND_N_POINTS new points per round, each round
    #      doubling the distance of the side's end to the estimate (EXTEND_FACTOR=1; None: up to the bound),
    #      at most MAX_EXTEND_ROUNDS rounds (0: no extension)
    #   3. CI_N_POINTS points over each CI, widened by CI_MARGIN x CI half-width on each side (0: skip);
    #      repeated with twice as many values for the profiles with fewer than MIN_CI_POINTS points inside
    #      the CI, at most MAX_CI_ROUNDS rounds in total
    # False: only step 1 (run_profile_likelihood_all, as in test_2sp.py)
    ADAPTIVE = True
    MAX_EXTEND_ROUNDS = 5
    EXTEND_FACTOR = 1.0
    CI_N_POINTS = 2
    CI_MARGIN = 0.3
    MIN_CI_POINTS = 5
    MAX_CI_ROUNDS = 3
    # tools for existing profiles (see test_2sp.py)
    #   None: no extension, 'auto': EXTEND_AUTO, 'manual': EXTEND_MANUAL ({parameter: spec})
    EXTEND_MODE = None
    EXTEND_AUTO = {'auto': True, 'ci': {'n_points': 5, 'margin': 0.1, 'min_points': 5, 'max_rounds': 2}}
    EXTEND_MANUAL = {}             # e.g. {'k_01,00': {'ci': 15, 'margin': 0.5}}
    EXTEND = {None: None, 'auto': EXTEND_AUTO, 'manual': EXTEND_MANUAL}[EXTEND_MODE]
    EXTEND_N_POINTS = 5
    UPDATE_OPTIMUM = False         # new Result_calibration as optimum of the existing profiles
    POLISH_PROFILES = False        # False, True or a list of parameters
    # ================================================================
    if BACKEND != 'numpy':
        raise ValueError("BACKEND must be 'numpy': the JAX backend does not contain the S terms")
    if OBJECTIVE not in ('neg2logL', 'geomean', 'known_noise') or CALIBRATION_OBJECTIVE != OBJECTIVE:
        raise ValueError("OBJECTIVE must be 'neg2logL', 'geomean' or 'known_noise' and CALIBRATION_OBJECTIVE = OBJECTIVE")
    cost_func = cost_withS_Sterms
    run_profiles = pl.run_profile_likelihood_adaptive if ADAPTIVE else pl.run_profile_likelihood_all

    def adaptive_options(stop_factor):
        """Extra arguments of run_profile_likelihood_adaptive; extension walks stop as the main walks do."""
        if not ADAPTIVE:
            return {}
        return dict(max_extend_rounds=MAX_EXTEND_ROUNDS, extend_factor=EXTEND_FACTOR,
                    extend_n_points=EXTEND_N_POINTS, extend_stop_factor=stop_factor,
                    ci_n_points=CI_N_POINTS, ci_margin=CI_MARGIN,
                    min_ci_points=MIN_CI_POINTS, max_ci_rounds=MAX_CI_ROUNDS,
                    extend_n_restarts=2)  # multistart per new point (steps 2 and 3)

    data_path = data_folder(DESIGN, relnoise, NORMALIZE_TRUE_S)
    path = data_path.rstrip('/') + '_withS/'
    os.makedirs(path, exist_ok=True)
    start = time.time()

    # 1) in-silico data -------------------------------------------------
    print(f'Design {DESIGN!r}: temperatures {temps}, days {days_meas if days_meas is not None else "0..17"}, '
          f'data {data_path}, results {path}')
    if RUN_DATA or not os.path.exists(data_path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        os.makedirs(data_path, exist_ok=True)
        S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir=fm.data.SETUP_DIR)
        x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir=fm.data.SETUP_DIR)['x0'])[:len(temps), :n_cl]
        if len(x10) < len(temps):
            raise ValueError(f'design {DESIGN!r} has {len(temps)} experiments, the setup has initial values for {len(x10)}')
        param_ode_gen = PARAM_ODE_TRUE.copy()
        if NORMALIZE_TRUE_S:
            S_matrix_setup, x10, param_ode_gen, m_s = normalize_truth_max_s(S_matrix_setup, x10, param_ode_gen, n_cl)
            print(f'True parameters rescaled to max(S) = 1 (factor {m_s:g})')
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup, param_ode=param_ode_gen,
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=data_path,
            add_name=add_name, t_meas=days_meas)
    else:
        dfs = [pd.read_pickle(data_path + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
        _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
            dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    df_mibi, df_maldi, df_ngs = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    bact = list(df_maldi.index)
    print(f'Data: {len(exps)} experiments {exps}, media {media}, species {bact}')

    # calibration setup (as in test_2sp.py) + masks of the S groups ------
    calibr_presetup = {
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': workers,
        'output_path': path,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(df_mibi),
        'media': media,
    }
    print('Temperatures:', calibr_presetup['exp_temps'])
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    calibr_setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    calibr_setup['s_max_penalty'] = S_MAX_PENALTY
    calibr_setup['noise_model'] = NOISE_MODEL
    calibr_setup['s_term_masks'] = S_term_masks(calibr_setup)
    n_dof = (fm.pest.count_data_points_noise(calibr_setup['data_array'], NOISE_MODEL)
             + [int(m.sum()) for m in calibr_setup['s_term_masks']])
    group_names = ('NGS', 'MALDI', 'MiBi') + S_GROUPS
    print('Data points per group: ' + ', '.join(f'{k} {n}' for k, n in zip(group_names, n_dof)))

    def make_objective(name):
        """Aggregation function of an objective name (see OBJECTIVE) over [NGS, MALDI, MiBi, S_ngs, S_media]
        (+ penalty group)."""
        if name == 'neg2logL':
            agg = pl.GaussianNeg2LogLik(n_dof)
        elif name == 'geomean':
            agg = pl.GeometricMeanNeg2LogLik(n_dof)
        else:
            sigma = list(np.broadcast_to(KNOWN_SIGMA, 3)) + list(np.broadcast_to(KNOWN_SIGMA_S, len(S_GROUPS)))
            agg = pl.KnownNoiseNeg2LogLik(n_dof, sigma)
        return SMaxPenaltyNeg2LogLik(agg, S_MAX_PENALTY) if S_MAX_PENALTY else agg

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
        p = np.asarray(data['param_ode'], dtype=float)
        fm.output.json_dump(dict(data, s_x=p[-n_cl*n_media:].reshape((n_media, n_cl)),
                                 objective=CALIBRATION_OBJECTIVE, noise_model=NOISE_MODEL, backend=BACKEND,
                                 s_terms=list(S_GROUPS)), fname, dir=path)

    def print_groups(param_list, setup):
        """Sum of squared residuals and number of data points per residual group."""
        for lab, p_ in param_list:
            groups, _ = residuals_withS_Sterms(p_, setup)
            print(f'  {lab:8s}: ' + ', '.join(f'{k} SSR = {np.sum(r**2):.4g} (n = {n})'
                                              for k, r, n in zip(group_names, groups, n_dof))
                  + f', max S = {np.max(np.asarray(p_)[-n_cl*n_media:]):.3g}')

    def load_param(fname):
        """param_ode of a result JSON; results of the earlier version with s0 (last entry) lose it."""
        r = fm.output.read_from_json(fname, dir=path)
        p = np.array(r['param_ode'], dtype=float)
        if 's0' in r and len(p) == len(names) + 1:
            print(f'Note: {fname} is from the version with s0 (= {p[-1]:.4g}), s0 dropped')
            p = p[:-1]
        return p

    calibr_setup['aggregation_func'] = make_objective(CALIBRATION_OBJECTIVE)
    print(f'Objective of the optimizations: {CALIBRATION_OBJECTIVE} incl. S terms, noise model: {NOISE_MODEL}')
    names = make_param_names(exps, bact, media)
    param_true = true_parameter_vector(data_path, add_name, media, n_cl)
    assert len(param_true) == len(calibr_setup['param_bnds']) == len(names)
    print(f'Cost ({CALIBRATION_OBJECTIVE}) at the true parameters: {cost_func(param_true, calibr_setup, None):.6g}')

    # 2) global optimization ---------------------------------------------
    fm.pest.optimization.output_file = path + 'optimization_history1.csv'  # DE history into the results folder
    fixed_glob = resolve_fixed_values(FIX_PARAMS, names) if FIX_IN_GLOBAL else {}
    if RUN_GLOBAL:
        glob_setup = calibr_setup
        if fixed_glob:
            print_parameter_table(names, param_true, calibr_setup['param_bnds'], fixed_glob)
            glob_setup = fm.pest.apply_fixed_values(calibr_setup, fixed_glob, names=names)
        de_starts = []
        warm_file = data_path + f'Result_calibration{add_name}.json'
        if WARM_START and os.path.exists(warm_file):
            p_warm = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=data_path)['param_ode'], dtype=float)
            if len(p_warm) == len(names):
                lo, hi = np.array(glob_setup['param_bnds'], dtype=float).T
                p_warm = np.clip(p_warm, lo, hi)
                de_starts = [p_warm]
                print(f'Warm start from {warm_file}: cost = {cost_func(p_warm, glob_setup, None):.6g}')
        if GLOBAL_METHOD in ('de', 'de+multistart'):
            param_glob, cost_glob = fm.pest.calculate_model_params(cost_func, glob_setup, maxiter=500)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            de_starts = [param_glob] + de_starts
        if GLOBAL_METHOD in ('multistart', 'de+multistart'):
            param_glob, cost_glob = fm.pest.multistart_optimization(
                cost_func, glob_setup, n_starts=N_STARTS, extra_starts=de_starts, method=LOCAL_METHOD,
                backend=BACKEND,
                maxiter=200 if LOCAL_METHOD == 'least_squares' else 500, n_jobs=n_jobs,
                history_file=path + 'optimization_history_multistart.csv')
        elif GLOBAL_METHOD != 'de':
            raise ValueError("GLOBAL_METHOD must be 'de', 'multistart' or 'de+multistart'")
        print(f'{(time.time() - start) / 60.:.1f} min (global), cost = {cost_glob:.6g}')
        save_result(f'Result_calibration{add_name}_global.json',
                    {'param_ode': param_glob.astype(list), 'T_x': T_x, 'cost': cost_glob,
                     'fixed_params': fm.pest.fixed_values_record(fixed_glob, names), 's_max_penalty': S_MAX_PENALTY})
    else:
        param_glob = load_param(f'Result_calibration{add_name}_global.json')
    # with S_MAX_PENALTY the estimates correspond to the truth rescaled to max(S) = 1 (extra column)
    param_true_rs = (rescale_truth_max_s(param_true, n_cl, len(exps), n_media)
                     if S_MAX_PENALTY and not NORMALIZE_TRUE_S else None)
    print_comparison(names, param_glob, param_true, calibr_setup['param_bnds'], 'Global optimization vs. truth:',
                     true_rescaled=param_true_rs)

    # 3) local refinement ------------------------------------------------
    if RUN_LOCAL:
        fixed = resolve_fixed_values(FIX_PARAMS, names)
        print_parameter_table(names, param_glob, calibr_setup['param_bnds'], fixed)
        param_opt, cost_opt = fm.pest.local_optimization(
            cost_func, param_glob, calibr_setup, fixed_values=fixed,
            maxiter=400 if LOCAL_METHOD == 'least_squares' else 1000, n_restarts=3, jitter_frac=0.1,
            n_jobs=n_jobs, method=LOCAL_METHOD, backend=BACKEND, history_file=path + 'optimization_history_local.csv')
        print(f'{(time.time() - start) / 60.:.1f} min (global + local), cost = {cost_opt:.6g}')
        save_result(f'Result_calibration{add_name}.json',
                    {'param_ode': param_opt.astype(list), 'T_x': T_x, 'cost': cost_opt,
                     'fixed_params': fm.pest.fixed_values_record(fixed, names), 's_max_penalty': S_MAX_PENALTY})
        copy_result_to_all_profiles(path, add_name)  # new optimum into existing profile JSON files
    else:
        param_opt = load_param(f'Result_calibration{add_name}.json')
    result_local = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)
    fixed = fixed_values_from_result(result_local)
    result_objective = result_local.get('objective', 'calibration')
    result_noise_model = result_local.get('noise_model', 'absolute')
    result_backend = result_local.get('backend', 'numpy')
    print_comparison(names, param_opt, param_true, calibr_setup['param_bnds'], 'Local refinement vs. truth:',
                     true_rescaled=param_true_rs)
    print_groups((('truth', param_true), ('estimate', param_opt)), calibr_setup)

    # 4) profile likelihood ----------------------------------------------
    if RUN_PROFILE:
        out_path = path + 'profile_likelihood/'
        os.makedirs(out_path, exist_ok=True)
        pl_setup = dict(calibr_setup)
        if fixed:  # parameters fixed in the refinement stay fixed (not profiled, not re-fitted)
            pl_setup, param_opt = fm.pest.apply_fixed_values(pl_setup, fixed, param=param_opt, names=names)
        pl_setup['aggregation_func'] = make_objective(OBJECTIVE)
        same_optimum = (({result_objective, OBJECTIVE} <= {'geomean', 'calibration'} or result_objective == OBJECTIVE)
                        and result_noise_model == NOISE_MODEL and result_backend == BACKEND)
        if not same_optimum:
            print(f'Result_calibration was optimized with {result_objective!r} / noise model {result_noise_model!r}, '
                  f'the profiles use {OBJECTIVE!r} / {NOISE_MODEL!r}: the estimate is re-fitted first (refine)')
        print(f'Profile objective ({OBJECTIVE}) at the truth: {cost_func(param_true, pl_setup, None):.6g}, '
              f'at the estimate: {cost_func(param_opt, pl_setup, None):.6g}')

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
            pl.update_profile_optimum(cost_func, pl_setup, out_csv, param_opt, backend=BACKEND,
                                      true_params=param_true if SHOW_TRUE_VALUE else None,
                                      plot_path=out_path + f'profile_likelihood{tag}.png')
        elif EXTEND is not None:
            print('Parameter indices:')
            for i in profile_indices:
                print(f'  {i:3d}  {names[i]:<28} estimate = {param_opt[i]:.6g}, bounds {calibr_setup["param_bnds"][i]}')
            pl.extend_profile_likelihood(
                cost_func, pl_setup, out_csv, EXTEND,
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
            run_profiles(
                cost_func, param_opt, pl_setup,
                **adaptive_options(stop_factor=1.5),
                span=0.3, n_points=15, method='local',  # coarse grid: the adaptive steps 2-3 add points where needed
                n_jobs=n_jobs,
                out_csv=out_csv,
                plot_path=out_path + f'profile_likelihood{tag}.png',
                param_names=names,
                n_restarts=1, jitter_frac=0.05,  # walks start from the neighbouring point: 1 start suffices
                profile_indices=profile_indices,
                fixed_indices=fixed_indices,
                refine=FIX_S == 'true' or not same_optimum,  # re-fit first if the optimum differs (FIX_S='true', objective)
                stop_factor=1.5,                 # stop each side once Delta > 1.5 x threshold
                minimize_options=profile_options,
                true_params=param_true if SHOW_TRUE_VALUE else None,
                verbose=1,
                restart_on_better=False,
                resume=True,
            )
        if POLISH_PROFILES:
            polish_idx = (None if POLISH_PROFILES is True
                          else sorted(resolve_fixed_values({k: 0. for k in POLISH_PROFILES}, names)))
            pl.polish_profile_likelihood(cost_func, pl_setup, out_csv, profile_indices=polish_idx,
                                         n_jobs=n_jobs, fixed_indices=fixed_indices,
                                         minimize_options=profile_options,
                                         true_params=param_true if SHOW_TRUE_VALUE else None,
                                         plot_path=out_path + f'profile_likelihood{tag}.png')
        pl.plot_profile_likelihood_individual(out_csv, out_dir=out_path + 'individual', file_ext='pdf',
                                              true_params=param_true if SHOW_TRUE_VALUE else None)
    print(f'{(time.time() - start) / 60.:.1f} min in total')

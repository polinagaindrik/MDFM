"""
Test case: 2-species model, S matrix estimated separately
================================================================================

Two-step pipeline of examples/main_ZL2030_woS.py on the in-silico data of test_2sp.py,
with a profile likelihood after each step:

  1) in-silico data: as test_2sp.py (same folder by default, so both scripts fit the same data)
  2) S estimation from the data alone (media_matrix.S_squared_difference, objective of
     calc_Smatrix = S_cost_func2): differential evolution   -> S_matrix<add_name>_global.json
     + local refinement (L-BFGS-B multistart)               -> S_matrix<add_name>.json
  3) profile likelihood of the S-step parameters [S (media sorted) | s0]
                                                            -> profile_likelihood_S/...
  4) estimation of x0 + ODE parameters with S fixed at the step-2 estimate (cost_direct):
     global (differential evolution) + local refinement     -> Result_calibration<add_name>.json
  5) profile likelihood of x0 + ODE parameters (S fixed)    -> profile_likelihood/...

S-step parameters: S_squared_difference uses the raw vector [S | s0]; calc_Smatrix then
multiplies one S row by s0 (row 'sel1' if s0 <= 1, else row 'gen1'). The profiles are taken
in the raw parameters (that is what the cost sees); the true raw vector is [S_true | 1].
The S step only uses ratios (MALDI fractions, NGS, MiBi relation between the media), so S is
determined up to a common factor c. The model has the approximate symmetry (S, x0, N_1, k_ij) ->
(c S, x0 - log10 c, N_1 - log10 c, c k_ij) (not exact: the active compartment always starts at 1 cell),
so steps 4-5 are compared with the truth rescaled to the S scale of the estimate (rescale_truth) as a
reference; the printed costs show how close that reference is to the optimum.
The S cost is no likelihood: S_OBJECTIVE turns its two residual groups (NGS/MALDI relation,
MiBi/MALDI media scaling) into a Gaussian -2 log L, so the chi2 threshold is approximate.

Each stage can be switched off (RUN_*) and is then read from its saved result.
Run from the repository root:
    python model_paper/test_2sp_sepS.py
"""
import os
import sys
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from profile_likelihood_model_paper import make_param_names, exp_temps_from_columns
from test_2sp import N_CL, PARAM_ODE_TRUE, GEN_MEDIA, print_comparison

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# S step in the cost_func(param, calibr_setup, jac_spasity) form of the profile likelihood engine
# ----------------------------------------------------------------------

def cost_S(param, calibr_setup, jac_spasity=None):
    """S-matrix cost for param = [S (media sorted, flattened) | s0]; residual groups [ngs term, s0 term]."""
    sq_ngs, sq_s0 = fm.pest.S_squared_difference(np.asarray(param, dtype=float), calibr_setup['dfs'],
                                                 calibr_setup['data_array'], calibr_setup['T_x'])
    return calibr_setup['aggregation_func']([sq_ngs, sq_s0])


def S_logsumexp(J_vect):
    """Aggregation of fm.pest.S_cost_func2 (cost_S with it equals S_cost_func2)."""
    a = 0.1
    num_meas = np.sum([np.count_nonzero(~np.isnan(J)) for J in J_vect])
    return np.log(np.nansum([np.nansum(np.exp(a*J)) for J in J_vect])/num_meas)/a


def S_from_raw(param_S, n_cl):
    """Final S matrix from the raw vector [S | s0], as in fm.pest.calc_Smatrix."""
    s0 = param_S[-1]
    s_x = np.array(param_S[:-1], dtype=float).reshape(-1, n_cl)
    if s0 <= 1:
        s_x[1] = s0*s_x[1]
    else:
        s_x[0] = s0*s_x[0]
    return s_x


def S_n_data(S_setup, param_S):
    """Number of measured entries in each residual group of cost_S (does not depend on param)."""
    J_vect = fm.pest.S_squared_difference(np.asarray(param_S, dtype=float), S_setup['dfs'],
                                          S_setup['data_array'], S_setup['T_x'])
    return [np.count_nonzero(np.isfinite(J)) for J in J_vect]


def residuals_direct(param, calibr_setup, jac_spasity=None):
    """Signed residuals of cost_direct (S from calibr_setup['s_x']), for least_squares."""
    return fm.pest.residuals_withS(np.concatenate([np.asarray(param, dtype=float), np.ravel(calibr_setup['s_x'])]),
                                   calibr_setup, jac_spasity)


def S_scale_factor(s_x, s_true):
    """Least-squares factor c with s_x ~ c * s_true (the S step only determines S up to such a factor)."""
    s_x, s_true = np.ravel(s_x), np.ravel(s_true)
    return float(np.dot(s_x, s_true) / np.dot(s_true, s_true))


def rescale_truth(param_true, c, n_cl, n_exps):
    """
    Approximately equivalent true [x0 | ODE] vector for S = c * S_true. The observables depend on S*x
    and on x through N_1 and k_ij*x, so (S, x, N_1, k_ij) -> (c S, x/c, N_1/c, c k_ij) gives nearly the
    same data: x0 and N_1 (log10) shift by -log10(c), k_ij is multiplied by c. Not exact, because the
    initial active compartment is fixed at 1 cell (C0 in sq_diff_oneexp) and does not scale with x0.
    """
    p = np.array(param_true, dtype=float)
    n_x0 = n_cl * n_exps
    p[:n_x0] -= np.log10(c)
    p[n_x0 + 4*n_cl] -= np.log10(c)
    p[n_x0 + 4*n_cl + 2:] *= c
    return p


def load_true(path, add_name, media_sorted, n_cl):
    """True S (media sorted) and true [x0 | ODE] vector from Result_real<add_name>.json."""
    tr = fm.output.read_from_json(f'Result_real{add_name}.json', dir=path)
    s_true = np.array(tr['s_x'], dtype=float).reshape(len(GEN_MEDIA), n_cl)
    s_true = np.array([s_true[GEN_MEDIA.index(m)] for m in media_sorted])
    return s_true, np.ravel(tr['param_ode']).astype(float)


if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    data_path = 'model_paper/out/test_2sp/'       # in-silico data (shared with test_2sp.py)
    path = 'model_paper/out/test_2sp_sepS/'       # results of this script
    n_cl = N_CL
    n_media = 2
    relnoise = 0.1
    temps = [2., 6., 10.]
    ntr = 1
    add_name = f'_{n_cl}dim_{n_media}media_test'
    workers = 20                   # differential evolution
    n_jobs = 20                    # local multistart / profile likelihood processes

    RUN_DATA = False               # 1) generate the in-silico data (also done if no data in data_path)
    RUN_S = True                   # 2a) S estimation, differential evolution
    RUN_S_LOCAL = True             # 2b) S estimation, local refinement (L-BFGS-B)
    RUN_PROFILE_S = True          # 3) profile likelihood of S
    RUN_GLOBAL = True              # 4a) differential evolution, S fixed
    RUN_LOCAL = True               # 4b) local refinement, S fixed
    RUN_PROFILE = True             # 5) profile likelihood of x0 + ODE parameters, S fixed

    # S profile objective: 'neg2logL' (data groups weighted by their number of points) or
    # 'geomean' (equal weight per group); the S estimate itself uses S_cost_func2 (logsumexp),
    # so the S optimum is re-fitted with the profile objective first
    S_OBJECTIVE = 'neg2logL'
    S_SPAN, S_N_POINTS = 1., 30    # cheap cost (no ODE): whole grid, no early stop
    S_LOCAL_RESTARTS = 8           # local S fits: start 0 = DE result, the others jittered around it

    # optimizer settings for step 4 (see test_2sp.py)
    GLOBAL_METHOD = 'de'           # 'de', 'multistart' or 'de+multistart'
    N_STARTS = 20
    LOCAL_METHOD = 'least_squares'      # 'L-BFGS-B' or 'least_squares'

    # profile likelihood settings for step 5 (see test_2sp.py)
    PROFILE = 'all'                # 'x0', 'ode' or 'all' (= x0 + ode)
    FIX_X0 = False                 # True: x0 held at the estimate
    OBJECTIVE = 'neg2logL'          # 'neg2logL', 'geomean' or 'calibration'
    SHOW_TRUE_VALUE = True
    # ================================================================

    os.makedirs(path, exist_ok=True)
    start = time.time()

    # 1) in-silico data -------------------------------------------------
    if RUN_DATA or not os.path.exists(data_path + f'dataframe_mibi{add_name}_preprocessed.pkl'):
        S_matrix_setup = fm.output.read_from_json('Media_matrix_S_paper.json', dir='model_paper/out/')
        x10 = np.array(fm.output.read_from_json('Initial_values_x0_paper.json', dir='model_paper/out/')['x0'])[:len(temps), :n_cl]
        dfs, bact_all, T_x, s_x_predefined = fm.data.prepare_insilico_data(
            fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup, param_ode=PARAM_ODE_TRUE.copy(),
            x10=x10, inhib=True, noise=0., rel_noise=relnoise, cutoff=0., cutoff_prop=0., path=data_path,
            add_name=add_name)
    else:
        dfs = [pd.read_pickle(data_path + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
        _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
            dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    df_mibi, df_maldi, df_ngs = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    exps = sorted(list(set([s.split('_')[0] for s in df_mibi.columns])))
    bact = list(df_maldi.index)
    data_array = fm.dtf.extract_observables_from_df(dfs)
    print(f'Data: {len(exps)} experiments {exps}, media {media}, species {bact}')

    names_all = make_param_names(exps, bact, media)
    s_true, param_true = load_true(data_path, add_name, media, n_cl)
    names_S = names_all[-n_media*n_cl:] + [r'$s_0$']
    names = names_all[:-n_media*n_cl]
    S_raw_true = np.concatenate([s_true.ravel(), [1.]])

    # 2) S estimation (calc_Smatrix) -------------------------------------
    S_bnds = [(0.05, 1.) if np.isnan(s) else (float(s), float(s)) for s in np.ravel(s_x_predefined)]
    S_setup = {
        'param_bnds': tuple(S_bnds + [(0.2, 5.)]),
        'dfs': dfs,
        'data_array': data_array,
        'T_x': T_x,
        'aggregation_func': S_logsumexp,
    }
    if RUN_S:
        fm.pest.media_matrix.output_file = path + 'optimization_history_Smatrix.csv'
        with open(fm.pest.media_matrix.output_file, 'w') as f:
            f.write('iteration,' + ''.join(f's{i // n_cl}{i % n_cl},' for i in range(len(S_setup['param_bnds']))) + 'cost\n')
        res_S = fm.pest.optimization_func_Smatrix(cost_S, S_setup['param_bnds'], args=(S_setup, None), workers=workers)
        param_S_glob, cost_S_glob = np.array(res_S.x), float(res_S.fun)
        print(f'{(time.time() - start) / 60.:.1f} min (S differential evolution), cost = {cost_S_glob:.6g}')
        fm.output.json_dump({'s_x': S_from_raw(param_S_glob, n_cl).astype(list), 'T_x': T_x,
                             'param_S_raw': param_S_glob.astype(list), 'cost': cost_S_glob, 'media': media},
                            f'S_matrix{add_name}_global.json', dir=path)
    else:
        param_S_glob = np.array(fm.output.read_from_json(f'S_matrix{add_name}_global.json', dir=path)['param_S_raw'])
    print_comparison(names_S, param_S_glob, S_raw_true, S_setup['param_bnds'], 'S differential evolution (raw [S | s0]) vs. truth:')

    # local refinement of S (L-BFGS-B, same objective S_cost_func2); the cost is cheap (no ODE),
    # so the starts run in parallel, one per process, instead of a parallel gradient
    if RUN_S_LOCAL:
        param_S, cost_S_opt = fm.pest.local_optimization(cost_S, param_S_glob, S_setup, maxiter=2000, gtol=1e-12,
                                                         n_restarts=S_LOCAL_RESTARTS, jitter_frac=0.05,
                                                         n_jobs=min(n_jobs, S_LOCAL_RESTARTS), print_every=0,
                                                         history_file=path + 'optimization_history_Smatrix_local.csv')
        s_x = S_from_raw(param_S, n_cl)
        print(f'{(time.time() - start) / 60.:.1f} min (S estimation, global + local), cost = {cost_S_opt:.6g}')
        fm.output.json_dump({'s_x': s_x.astype(list), 'T_x': T_x, 'param_S_raw': param_S.astype(list),
                             'cost': cost_S_opt, 'media': media}, f'S_matrix{add_name}.json', dir=path)
    else:
        res = fm.output.read_from_json(f'S_matrix{add_name}.json', dir=path)
        param_S, s_x = np.array(res['param_S_raw']), np.array(res['s_x'])
    print(f'S cost (S_cost_func2) at the truth: {cost_S(S_raw_true, S_setup):.6g}, '
          f'at the estimate: {cost_S(param_S, S_setup):.6g}')
    print_comparison(names_S, param_S, S_raw_true, S_setup['param_bnds'], 'S step (raw [S | s0]) vs. truth:')
    print_comparison(names_all[-n_media*n_cl:], s_x.ravel(), s_true.ravel(),
                     [(0., np.inf)]*(n_media*n_cl), 'Final S (s0 applied) vs. truth:')
    c_S = S_scale_factor(s_x, s_true)
    print_comparison(names_all[-n_media*n_cl:], s_x.ravel() / c_S, s_true.ravel(), [(0., np.inf)]*(n_media*n_cl),
                     f'Final S / c vs. truth (S is determined up to a factor only; c = {c_S:.4g}):')
    # truth in the S scale of the estimate: reference for steps 4 and 5
    param_true_c = rescale_truth(param_true, c_S, n_cl, len(exps))

    # 3) profile likelihood of S -----------------------------------------
    if RUN_PROFILE_S:
        out_path_S = path + 'profile_likelihood_S/'
        os.makedirs(out_path_S, exist_ok=True)
        pl_setup_S = dict(S_setup)
        pl_setup_S['aggregation_func'] = {'neg2logL': pl.GaussianNeg2LogLik,
                                          'geomean': pl.GeometricMeanNeg2LogLik}[S_OBJECTIVE](S_n_data(S_setup, param_S))
        tag_S = f'{add_name}_S_{S_OBJECTIVE}'
        out_csv_S = out_path_S + f'profile_likelihood_results{tag_S}.csv'
        pl.run_profile_likelihood_all(
            cost_S, param_S, pl_setup_S,
            span=S_SPAN, n_points=S_N_POINTS, method='local',
            n_jobs=n_jobs,
            out_csv=out_csv_S,
            plot_path=out_path_S + f'profile_likelihood{tag_S}.png',
            param_names=names_S,
            n_restarts=3, jitter_frac=0.05,
            refine=True,                   # S was estimated with S_cost_func2, not with the profile objective
            stop_factor=None,
            minimize_options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10},
            true_params=S_raw_true if SHOW_TRUE_VALUE else None,
            verbose=1,
            resume=True,
        )
        pl.plot_profile_likelihood_individual(out_csv_S, out_dir=out_path_S + 'individual', file_ext='pdf',
                                              true_params=S_raw_true if SHOW_TRUE_VALUE else None)
        print(f'{(time.time() - start) / 60.:.1f} min (S profile likelihood)')

    # 4) x0 + ODE parameters with S fixed (cost_direct) ------------------
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
    calibr_setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=s_x)
    calibr_setup['data_array'] = data_array
    assert len(param_true) == len(calibr_setup['param_bnds']) == len(names)
    print(f'Cost at the true x0 + ODE parameters with the true S: '
          f'{fm.pest.cost_direct(param_true, dict(calibr_setup, s_x=s_true), None):.6g}, rescaled to the estimated S '
          f'(c = {c_S:.4g}): {fm.pest.cost_direct(param_true_c, dict(calibr_setup, s_x=c_S*s_true), None):.6g}, '
          f'with the estimated S: {fm.pest.cost_direct(param_true_c, calibr_setup, None):.6g}')

    local_kw = dict(maxiter=200 if LOCAL_METHOD == 'least_squares' else 1000, method=LOCAL_METHOD,
                    residual_func=residuals_direct if LOCAL_METHOD == 'least_squares' else None)
    fm.pest.optimization.output_file = path + 'optimization_history1.csv'
    if RUN_GLOBAL:
        de_starts = []
        if GLOBAL_METHOD in ('de', 'de+multistart'):
            param_glob, cost_glob = fm.pest.calculate_model_params(fm.pest.cost_direct, calibr_setup)
            print(f'{(time.time() - start) / 60.:.1f} min (differential evolution), cost = {cost_glob:.6g}')
            de_starts = [param_glob]
        if GLOBAL_METHOD in ('multistart', 'de+multistart'):
            param_glob, cost_glob = fm.pest.multistart_optimization(
                fm.pest.cost_direct, calibr_setup, n_starts=N_STARTS, extra_starts=de_starts, n_jobs=n_jobs,
                history_file=path + 'optimization_history_multistart.csv', **local_kw)
        elif GLOBAL_METHOD != 'de':
            raise ValueError("GLOBAL_METHOD must be 'de', 'multistart' or 'de+multistart'")
        fm.output.json_dump({'param_ode': param_glob.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_glob},
                            f'Result_calibration{add_name}_global.json', dir=path)
    else:
        param_glob = np.array(fm.output.read_from_json(f'Result_calibration{add_name}_global.json', dir=path)['param_ode'])
    print_comparison(names, param_glob, param_true_c, calibr_setup['param_bnds'], 'Global optimization (S fixed) vs. rescaled truth:')

    if RUN_LOCAL:
        param_opt, cost_opt = fm.pest.local_optimization(fm.pest.cost_direct, param_glob, calibr_setup,
                                                         n_restarts=3, jitter_frac=0.05, n_jobs=n_jobs,
                                                         history_file=path + 'optimization_history_local.csv',
                                                         **local_kw)
        print(f'{(time.time() - start) / 60.:.1f} min (S + global + local), cost = {cost_opt:.6g}')
        fm.output.json_dump({'param_ode': param_opt.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_opt},
                            f'Result_calibration{add_name}.json', dir=path)
        fm.plotting.plot_optimization_result(np.array(param_opt), calibr_setup, np.linspace(0, 17, 100),
                                             path=path, add_name=f'{add_name}_calibration')
    else:
        param_opt = np.array(fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path)['param_ode'])
    print_comparison(names, param_opt, param_true_c, calibr_setup['param_bnds'], 'Local refinement (S fixed) vs. rescaled truth:')

    # 5) profile likelihood of x0 + ODE parameters (S fixed) -------------
    if RUN_PROFILE:
        out_path = path + 'profile_likelihood/'
        os.makedirs(out_path, exist_ok=True)
        pl_setup = dict(calibr_setup)
        pl_setup['aggregation_func'] = {
            'neg2logL': pl.make_gaussian_neg2loglik(data_array),
            'geomean': pl.make_geomean_neg2loglik(data_array),
            'calibration': fm.pest.cost_sum_and_geometric_mean}[OBJECTIVE]

        n_x0 = n_cl * len(exps)
        idx_x0 = list(range(n_x0))
        profile_indices = {'x0': idx_x0, 'ode': list(range(n_x0, len(param_opt))), 'all': None}[PROFILE]
        if FIX_X0 and PROFILE in ('x0', 'all'):
            raise ValueError('PROFILE=' + repr(PROFILE) + ' profiles x0, so FIX_X0 must be False')
        fixed_indices = idx_x0 if FIX_X0 else []
        if profile_indices is None:
            profile_indices = list(range(len(param_opt)))
        profile_indices = [i for i in profile_indices if i not in fixed_indices]
        tag = f"{add_name}_{PROFILE}_{OBJECTIVE}" + ('_fixx0' if FIX_X0 else '')
        out_csv = out_path + f'profile_likelihood_results{tag}.csv'

        profile_options = ({'method': 'least_squares', 'maxiter': 200, 'residual_func': residuals_direct}
                           if LOCAL_METHOD == 'least_squares' else {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        pl.run_profile_likelihood_all(
            fm.pest.cost_direct, param_opt, pl_setup,
            span=3, n_points=12, method='local',
            n_jobs=n_jobs,
            out_csv=out_csv,
            plot_path=out_path + f'profile_likelihood{tag}.png',
            param_names=names,
            n_restarts=1, jitter_frac=0.05,
            profile_indices=profile_indices,
            fixed_indices=fixed_indices,
            refine=OBJECTIVE == 'neg2logL',  # 'geomean'/'calibration' have the calibration optimum
            stop_factor=3,
            minimize_options=profile_options,
            true_params=param_true_c if SHOW_TRUE_VALUE else None,
            verbose=1,
            resume=True,
        )
        pl.plot_profile_likelihood_individual(out_csv, out_dir=out_path + 'individual', file_ext='pdf',
                                              true_params=param_true_c if SHOW_TRUE_VALUE else None)
    print(f'{(time.time() - start) / 60.:.1f} min in total')

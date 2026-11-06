"""
Multistart fit of the 12-species in-silico model to the ZL2030 data
===================================================================

Fits the ODE model and bounds of the in-silico case (fusion_model2, define_calibr_setup_insilico) to the real data, to
compare with the parameter set used for the in-silico studies (out/zl2030/exp_model/calibration/Result_calibration_zl2030_copy.json).

Differences to that earlier fit (differential evolution with the ZL2030 bounds and cost_sum_and_geometric_mean):
  - objective: -2 log L with the relative noise model and the max S = 1 penalty.
    The real MALDI and NGS data have no fixed number of counts per sample, so the multinomial and Dirichlet-multinomial
    likelihoods of the in-silico fits (which need one) are not used here.
  - bounds: in-silico bounds, e.g. lambda_exp in (0, 3) and 'Others' may grow (lower bound alpha_0 = 0.001).

Start 0 is the earlier parameter set (clipped to the bounds), the other starts are Latin hypercube points. Comparing
start 0 with the earlier set shows the effect of objective and bounds alone.

Output: out/zl2030/multistart/
    Result_calibration_zl2030_multistart.json   best parameters
    comparison_parameters.csv                   earlier vs. best ODE and S parameters, range over the best starts
    start_costs.csv                             best cost of every start
Run from anywhere:
    python examples/zl2030/main_zl2030_multistart.py
To stop early and keep the best points: touch <output>/optimization_history_multistart.stop

Second batch, more independent starts with the inhibition terms sampled on a log scale (no start from the earlier set):
    python examples/zl2030/main_zl2030_multistart.py --log-k
The in-silico bounds of k_ij start at 0, so the Latin hypercube samples them linearly (median about 10, against 1.4 in
the earlier set). Here they are sampled in K_START_RANGE instead. Output: out/zl2030/multistart_logk/
"""
import glob
import json
import os
import re
import sys
import time
# CPU only, one thread per process: the N_JOBS processes share one GPU otherwise and run out of its memory.
# Before the imports below, the spawned worker processes inherit these.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.parameter_estimation import profile_likelihood as pl
from fusion_model.parameter_estimation.calibration_utils import make_param_names
from fusion_model.parameter_estimation.jax_backend import cost_withS_jax

import numpy as np
import pandas as pd

N_STARTS = 95                 # Latin hypercube starts; the earlier parameter set comes on top
N_JOBS = os.cpu_count()       # all cores, one start per core at a time
MAX_ITER = 500                # residual evaluations per least-squares solve
MAX_MINUTES = 240             # limit per start
SEED = 0
S_MAX_PENALTY = 100           # penalty so that the largest media coefficient S is 1
NOISE_MODEL = 'relative'
LOG_K = '--log-k' in sys.argv
N_STARTS_LOG_K = 48           # --log-k: Latin hypercube starts
K_START_RANGE = (0.02, 20.)   # --log-k: k_ij of the start points, log10 uniform; the fit bounds stay (0, 20)
LAMBDA_EXP_MAX = 3.           # upper bound of lambda_exp
NEAR_BEST = 3.84              # starts within this cost of the best count as "near best" (chi2, 1 dof, 95 %)

CUTOFF, CUTOFF_PROP = 0.1, 0.001    # gives the 11 prevalent species + Others of the earlier fit
EXPS_NGS = [f'V{i:02d}' for i in range(1, 13)] + [f'V{i:02d}' for i in range(15, 21)] + ['V22']
EXPS_MALDI = [f'V{i:02d}' for i in range(1, 20)] + ['V22']
EXPS = [f'V{i:02d}' for i in range(1, 10)] + ['V11', 'V12'] + [f'V{i:02d}' for i in range(16, 20)]
N_MEDIA = 2

# the data readers use paths relative to the MDFM repository (experiments/)
MDFM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(fm.__file__)))
OUT = os.path.join(MDFM_ROOT, 'out', 'zl2030', 'multistart_logk' if LOG_K else 'multistart', '')
EARLIER = os.path.join(MDFM_ROOT, 'out', 'zl2030', 'exp_model', 'calibration', 'Result_calibration_zl2030_copy.json')


def best_of_start(fname):
    """(best cost, parameters) of one start from its history file."""
    h = pd.read_csv(fname)
    h = h.loc[:, ~h.columns.str.startswith('Unnamed')]
    i = h['cost'].idxmin()
    return float(h.loc[i, 'cost']), h.loc[i, [c for c in h.columns if re.fullmatch(r'p\d+', c)]].to_numpy(dtype=float)


def N_CL_X0(setup):
    """Number of initial values."""
    return setup['n_cl'] * len(setup['exps'])


def log_k_starts(param_bnds, n_starts, n_x0, n_cl, seed=SEED):
    """Latin hypercube starts with the free k_ij replaced by a Latin hypercube in K_START_RANGE on a log10 scale."""
    from fusion_model.parameter_estimation.least_squares_fit import latin_hypercube_starts
    starts = latin_hypercube_starts(param_bnds, n_starts, seed=seed)
    lo, hi = np.array(param_bnds, dtype=float).T
    i0 = n_x0 + 4 * n_cl + 2
    idx = [i for i in range(i0, i0 + n_cl * n_cl) if hi[i] > lo[i]]
    rng = np.random.default_rng(seed + 1)
    u = (np.argsort(rng.random((n_starts, len(idx))), axis=0) + rng.random((n_starts, len(idx)))) / n_starts
    k = 10.**(np.log10(K_START_RANGE[0]) + u * (np.log10(K_START_RANGE[1]) - np.log10(K_START_RANGE[0])))
    for s, row in zip(starts, k):
        s[idx] = row
    return starts


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.chdir(MDFM_ROOT)
    start = time.time()

    temps = fm.data.EXP_TEMPS
    dfs, bact, T_x, s_x_predefined = fm.data.prepare_ZL2030_data(EXPS, EXPS_MALDI, EXPS_NGS, cutoff=CUTOFF,
                                                                 cutoff_prop=CUTOFF_PROP, path=OUT, add_name='_zl2030')
    bact = list(dfs[1].index)
    media = sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns))
    n_cl = len(bact)
    print(f'{len(EXPS)} experiments, temperatures { {e: temps[e] for e in EXPS} }')
    print(f'{n_cl} species {bact}, media {media}', flush=True)
    if n_cl != 12:
        raise ValueError(f'expected 12 species, got {n_cl}')

    setup = fm.pest.define_calibr_setup_insilico(
        {'model': fm.mdl.fusion_model2, 'T_x': T_x, 'workers': 1, 'output_path': OUT, 'n_cl': n_cl, 'n_media': N_MEDIA,
         'dfs': dfs, 'exps': EXPS, 'exp_temps': {e: temps[e] for e in EXPS}, 'media': media},
        inhib=True, s_x_predefined=s_x_predefined, s_x=None, lambda_exp_max=LAMBDA_EXP_MAX)
    setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    setup['s_max_penalty'] = S_MAX_PENALTY
    setup['noise_model'] = NOISE_MODEL
    setup['maldi_n'] = None
    setup['ngs_dm'] = None
    setup['aggregation_func'] = pl.make_gaussian_neg2loglik(setup['data_array'], NOISE_MODEL)
    if LOG_K:  # same Jacobian as the default in the free columns (checked: difference ~1e-14), about 1.4 x faster
        setup['jax_options'] = {'structured_jacobian': True}
    names = make_param_names(EXPS, bact, media)

    # earlier parameter set, in the bounds of this fit
    earlier = np.array(fm.output.read_from_json(os.path.basename(EARLIER), dir=os.path.dirname(EARLIER) + '/')['param_ode'],
                       dtype=float)
    lo, hi = np.array(setup['param_bnds'], dtype=float).T
    if len(earlier) != len(lo):
        raise ValueError(f'earlier parameter set has {len(earlier)} entries, this fit {len(lo)}')
    for i in np.where((earlier < lo) | (earlier > hi))[0]:
        print(f'Earlier set outside the bounds, clipped: {names[i]} = {earlier[i]:.4g}, bounds ({lo[i]:.4g}, {hi[i]:.4g})')
    earlier_clipped = np.clip(earlier, lo, hi)
    print(f'{int(np.sum(hi > lo))} free parameters of {len(lo)}', flush=True)

    history = OUT + 'optimization_history_multistart.csv'
    best, cost_best = fm.pest.multistart_optimization(
        fm.pest.cost_withS, setup, n_starts=0 if LOG_K else N_STARTS, seed=SEED,
        extra_starts=log_k_starts(setup['param_bnds'], N_STARTS_LOG_K, N_CL_X0(setup), n_cl) if LOG_K else [earlier_clipped],
        method='least_squares',
        maxiter=MAX_ITER, n_jobs=N_JOBS, history_file=history, backend='jax', max_minutes=MAX_MINUTES)
    print(f'{(time.time() - start) / 60.:.1f} min (multistart), best cost = {cost_best:.6f}', flush=True)

    s_x = best[-n_cl * N_MEDIA:].reshape((N_MEDIA, n_cl))
    fm.output.json_dump({'param_ode': best.astype(list), 's_x': s_x, 'T_x': T_x, 'cost': cost_best,
                         'objective': 'neg2logL', 'noise_model': NOISE_MODEL, 'backend': 'jax', 'maldi_n': None,
                         'ngs_dm': None, 's_max_penalty': S_MAX_PENALTY, 'bact': bact, 'media': media},
                        'Result_calibration_zl2030_multistart.json', dir=OUT)

    # every start
    rows, params = [], {}
    for f in glob.glob(OUT + 'optimization_history_multistart_start*.csv'):
        r = int(re.search(r'_start(\d+)\.csv$', f).group(1))
        try:
            c, p = best_of_start(f)
        except (ValueError, KeyError, pd.errors.EmptyDataError) as err:
            print(f'Start {r}: no usable history ({err})')
            continue
        rows.append({'start': r, 'kind': 'earlier set' if r == 0 and not LOG_K else 'latin hypercube', 'cost': c,
                     'delta_to_best': c - cost_best})
        params[r] = p
    starts = pd.DataFrame(rows).sort_values('cost')
    starts.to_csv(OUT + 'start_costs.csv', index=False)
    near = [r for r in starts['start'] if starts.set_index('start').loc[r, 'delta_to_best'] <= NEAR_BEST]
    print(f'{len(near)} of {len(starts)} starts within {NEAR_BEST} of the best cost')
    print(starts.head(15).to_string(index=False), flush=True)

    # earlier vs. best, under both objectives
    cost_earlier = float(cost_withS_jax(earlier_clipped, setup))
    print(f'-2 log L (this fit):           earlier set {cost_earlier:.3f}, best {float(cost_withS_jax(best, setup)):.3f}')
    old_objective = dict(setup, aggregation_func=fm.pest.cost_sum_and_geometric_mean, noise_model='absolute',
                         s_max_penalty=None)
    print(f'Objective of the earlier fit:  earlier set {fm.pest.cost_withS(earlier_clipped, old_objective, None):.4f}, '
          f'best {fm.pest.cost_withS(best, old_objective, None):.4f}', flush=True)

    near_p = np.array([params[r] for r in near]) if near else best[None]
    pd.DataFrame({'name': names, 'earlier': earlier, 'best': best, 'start0_refit': params.get(0, np.full(len(best), np.nan)),
                  'near_best_min': near_p.min(axis=0), 'near_best_max': near_p.max(axis=0), 'lower_bound': lo,
                  'upper_bound': hi}).to_csv(OUT + 'comparison_parameters.csv', index=False)
    print(f'{(time.time() - start) / 60.:.1f} min in total, results in {OUT}')

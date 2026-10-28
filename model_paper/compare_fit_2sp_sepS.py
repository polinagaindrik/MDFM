"""
Goodness of fit: test_2sp.py (S fitted together with x0 + ODE parameters) vs. test_2sp_sepS.py
(S estimated separately from the data, then x0 + ODE parameters with S fixed) vs. test_2sp_withS.py
(all parameters together, with the S-step cost terms added to the objective)
================================================================================

All results (Result_calibration<add_name>.json of the output folders; withS: the _global result if the local
refinement has not been run yet; a missing withS result is skipped) and the truth are evaluated
with the same residuals (fm.pest.residuals_withS, noise model NOISE_MODEL, without the max(S) = 1 penalty
of test_2sp.py) on the same in-silico data. Per data type (NGS, MALDI, MiBi) and in total:
  n      : data points (degrees of freedom of the residuals, count_data_points_noise)
  SSR    : sum of squared residuals,  RMSE = sqrt(SSR / n)
  -2logL : sum_k n_k log(SSR_k / n_k) (Gaussian, noise variance estimated per data type; the 'neg2logL'
           objective of both scripts)
  AIC, BIC with k free parameters (sepS: x0 + ODE + the S step's free S entries)
In addition the S-step cost terms of test_2sp_withS.py (S_ngs, S_media: SSR and RMSE) are printed for every fit.
The joint fit minimizes -2logL over S as well, so at the optimum it can only be lower; the difference
shows how much the separate S estimate costs in fit quality.
Output: printed tables, fit_comparison<add_name>.csv / _per_exp.csv / .png in the sepS folder.
Run from the repository root:
    python model_paper/compare_fit_2sp_sepS.py
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from profile_likelihood_model_paper import exp_temps_from_columns, fixed_values_from_result
from test_2sp import N_CL, data_folder, true_parameter_vector
from test_2sp_withS import S_GROUPS, S_term_masks, _S_term_raw

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_TYPES = ['NGS', 'MALDI', 'MiBi']


def evaluate(param_full, setup, n_data):
    """Fit statistics of the full vector [x0 | ODE | S]: per-type and per-experiment table rows."""
    res, owner = fm.pest.residuals_withS(param_full, setup)
    rows, rows_exp = {}, []
    for name, r, o, n in zip(DATA_TYPES, res, owner, n_data):
        ssr = float(np.sum(r**2))
        rows[name] = {'n': n, 'SSR': ssr, 'RMSE': np.sqrt(ssr / n), '-2logL': n * np.log(ssr / n)}
        for i, exp in enumerate(setup['exps']):
            r_e = r[o == i]
            rows_exp.append({'exp': exp, 'data': name, 'SSR': float(np.sum(r_e**2)),
                             'RMSE': np.sqrt(np.mean(r_e**2)) if len(r_e) else np.nan})
    n_tot = sum(n_data)
    rows['total'] = {'n': n_tot, 'SSR': sum(v['SSR'] for v in rows.values()),
                     'RMSE': np.sqrt(sum(v['SSR'] for v in rows.values()) / n_tot),
                     '-2logL': sum(v['-2logL'] for v in rows.values())}
    return rows, rows_exp


def n_free(bnds, fixed):
    """Number of parameters that are neither fixed by their bounds nor by 'fixed_params'."""
    return sum(1 for i, b in enumerate(bnds) if b[1] > b[0] and i not in fixed)


if __name__ == "__main__":
    # ================================================================
    # Settings (as in test_2sp.py / test_2sp_sepS.py)
    # ================================================================
    DESIGN = 'rep3'
    relnoise = 0.1
    NORMALIZE_TRUE_S = False       # test_2sp.py setting (data folder suffix '_normS')
    NOISE_MODEL = 'relative'       # residuals of the comparison ('absolute' or 'relative')
    n_media = 2
    add_name = f'_{N_CL}dim_{n_media}media_test'
    # ================================================================
    path_joint = data_folder(DESIGN, relnoise, NORMALIZE_TRUE_S)
    path_sep = data_folder(DESIGN, relnoise).rstrip('/') + '_sepS/'
    path_withS = path_joint.rstrip('/') + '_withS/'
    n_cl = N_CL

    dfs = [pd.read_pickle(path_joint + f'dataframe_{k}{add_name}_preprocessed.pkl') for k in ['mibi', 'maldi', 'ngs']]
    _, _, T_x, s_x_predefined = fm.dtf.make_df_maldi_ngs_compatible(
        dfs[1], dfs[2], cutoff=0.001, media=sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns)))
    media = sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns))
    exps = sorted(set(s.split('_')[0] for s in dfs[0].columns))
    calibr_presetup = {
        'model': fm.mdl.fusion_model2,
        'T_x': T_x,
        'workers': 1,
        'output_path': path_sep,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs,
        'exps': exps,
        'exp_temps': exp_temps_from_columns(dfs[0]),
        'media': media,
    }
    setup = fm.pest.define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined, s_x=None)
    setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    setup['noise_model'] = NOISE_MODEL
    setup['s_max_penalty'] = None  # data residuals only
    n_data = fm.pest.count_data_points_noise(setup['data_array'], NOISE_MODEL)
    n_tot = sum(n_data)

    res_joint = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path_joint)
    res_sep = fm.output.read_from_json(f'Result_calibration{add_name}.json', dir=path_sep)
    res_withS = None
    for fname in (f'Result_calibration{add_name}.json', f'Result_calibration{add_name}_global.json'):
        if os.path.exists(path_withS + fname):
            res_withS = fm.output.read_from_json(fname, dir=path_withS)
            print(f'test_2sp_withS: {path_withS + fname}')
            break
    if res_withS is None:
        print(f'Note: no test_2sp_withS result in {path_withS}, skipped')
    for lab, r in (('test_2sp', res_joint), ('test_2sp_sepS', res_sep), ('test_2sp_withS', res_withS or {})):
        if not r:
            continue
        if r.get('noise_model', 'absolute') != NOISE_MODEL:
            print(f'Note: {lab} was fitted with noise model {r.get("noise_model", "absolute")!r}, '
                  f'the comparison uses {NOISE_MODEL!r}')

    # full vectors [x0 | ODE | S] (media sorted)
    param_joint = np.array(res_joint['param_ode'], dtype=float)
    param_sep = np.concatenate([np.array(res_sep['param_ode'], dtype=float), np.ravel(res_sep['s_x'])])
    param_true = true_parameter_vector(path_joint, add_name, media, n_cl)

    # free parameters: joint = all free entries of [x0 | ODE | S]; sepS = free x0 + ODE entries + S step S
    bnds = setup['param_bnds']
    n_S = n_media * n_cl
    k_joint = n_free(bnds, fixed_values_from_result(res_joint))
    k_sep = n_free(bnds[:-n_S], fixed_values_from_result(res_sep)) + int(np.sum(np.isnan(s_x_predefined)))
    fits = {'test_2sp (joint S)': (param_joint, k_joint),
            'test_2sp_sepS (separate S)': (param_sep, k_sep)}
    if res_withS is not None:  # vector [x0 | ODE | S] (results of the earlier version with s0: s0 dropped)
        p_w = np.array(res_withS['param_ode'], dtype=float)[:len(bnds)]
        fits['test_2sp_withS (joint S + S terms)'] = (p_w, n_free(bnds, fixed_values_from_result(res_withS)))
    fits['truth'] = (param_true, None)

    # S-step cost terms of test_2sp_withS.py for every fit
    setup['s_term_masks'] = S_term_masks(setup)
    rows_S = []
    for lab, (p, _) in fits.items():
        s_x = np.asarray(p, dtype=float)[-n_S:].reshape((n_media, n_cl))
        r_S = [r[m] for r, m in zip(_S_term_raw(s_x, setup), setup['s_term_masks'])]
        for g, r in zip(S_GROUPS, r_S):
            rows_S.append({'fit': lab, 'term': g, 'n': len(r), 'SSR': float(np.sum(r**2)),
                           'RMSE': float(np.sqrt(np.mean(r**2)))})
    df_S = pd.DataFrame(rows_S)

    rows, rows_exp = [], []
    for lab, (p, k) in fits.items():
        stats, stats_exp = evaluate(p, setup, n_data)
        for d, v in stats.items():
            row = {'fit': lab, 'data': d, **v}
            if d == 'total' and k is not None:
                row.update(k=k, AIC=v['-2logL'] + 2*k, BIC=v['-2logL'] + k*np.log(n_tot))
            rows.append(row)
        rows_exp += [{'fit': lab, **r} for r in stats_exp]
    df = pd.DataFrame(rows)
    df_exp = pd.DataFrame(rows_exp)

    # ------------------------------------------------------------------
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f'{v:.5g}')
    print(f'Data: {path_joint} ({len(exps)} experiments, noise model {NOISE_MODEL!r})')
    print(f'S joint: {np.round(res_joint["s_x"], 4).tolist()}, S sepS: {np.round(res_sep["s_x"], 4).tolist()}'
          + (f', S withS: {np.round(res_withS["s_x"], 4).tolist()}'
             if res_withS is not None else '') + '\n')
    print(df.to_string(index=False))
    tot = df[df.data == 'total'].set_index('fit')
    a, b = 'test_2sp (joint S)', 'test_2sp_sepS (separate S)'
    print(f'\nsepS - joint:  Delta(-2logL) = {tot.loc[b, "-2logL"] - tot.loc[a, "-2logL"]:.4g}, '
          f'Delta AIC = {tot.loc[b, "AIC"] - tot.loc[a, "AIC"]:.4g}, Delta BIC = {tot.loc[b, "BIC"] - tot.loc[a, "BIC"]:.4g}')
    c = 'test_2sp_withS (joint S + S terms)'
    if c in tot.index:
        print(f'withS - joint: Delta(-2logL) = {tot.loc[c, "-2logL"] - tot.loc[a, "-2logL"]:.4g}, '
              f'Delta AIC = {tot.loc[c, "AIC"] - tot.loc[a, "AIC"]:.4g}, Delta BIC = {tot.loc[c, "BIC"] - tot.loc[a, "BIC"]:.4g}')
    print('\nS-step cost terms (test_2sp_withS.py):')
    print(df_S.to_string(index=False))
    print('\nRMSE per experiment:')
    print(df_exp.pivot_table(index='exp', columns=['data', 'fit'], values='RMSE').to_string())

    out = path_sep + f'fit_comparison{add_name}'
    df.to_csv(out + '.csv', index=False)
    df_exp.to_csv(out + '_per_exp.csv', index=False)
    df_S.to_csv(out + '_Sterms.csv', index=False)

    # RMSE per data type and per experiment (one panel per data type)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), gridspec_kw={'width_ratios': [1, 2, 2, 2]})
    clrs = {a: 'tab:blue', b: 'tab:orange', c: 'tab:green', 'truth': 'tab:gray'}
    w = 0.8 / len(fits)
    shift = (len(fits) - 1) / 2
    for j, lab in enumerate(fits):
        sub = df[(df.fit == lab) & (df.data != 'total')]
        axes[0].bar(np.arange(len(DATA_TYPES)) + (j - shift)*w, sub['RMSE'], w, color=clrs[lab], label=lab)
        for ax, d in zip(axes[1:], DATA_TYPES):
            sub_e = df_exp[(df_exp.fit == lab) & (df_exp.data == d)]
            ax.bar(np.arange(len(exps)) + (j - shift)*w, sub_e['RMSE'], w, color=clrs[lab])
    axes[0].set_xticks(range(len(DATA_TYPES)), DATA_TYPES)
    axes[0].set_ylabel('RMSE of the residuals')
    axes[0].set_title('all experiments')
    for ax, d in zip(axes[1:], DATA_TYPES):
        ax.set_xticks(range(len(exps)), exps, rotation=90)
        ax.set_title(d)
    axes[0].legend(loc='upper center', bbox_to_anchor=(2.5, -0.2), ncol=len(fits), frameon=False)
    fig.tight_layout()
    fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved {out}.csv, {out}_per_exp.csv, {out}_Sterms.csv, {out}.png')

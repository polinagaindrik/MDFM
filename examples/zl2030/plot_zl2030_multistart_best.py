"""
Plot the best fit so far of main_zl2030_multistart.py, also while it is running (reads the history file of each start)
======================================================================================================================

Output: out/zl2030/multistart/best_so_far/
    overview_plate_counts_best_vs_earlier.png   plate counts of all experiments: data, best fit, earlier parameter set
    MiBi_<medium>_const_model.png               plate counts of all experiments per temperature, best fit (as plot_model_results.py)
    best/      per-experiment figures of the best fit (plate counts, MALDI, NGS, cells)
    earlier/   the same for the earlier parameter set
Run from anywhere:
    python examples/zl2030/plot_zl2030_multistart_best.py
"""
import glob
import os
import re
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fusion_model as fm
from fusion_model.plotting import plotting_templates
import main_zl2030_multistart as ms

import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

MEDIUM_COLOR = {'PC': '#2a78d6', 'MRS': '#eb6834'}
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def best_so_far():
    """(cost, start, full parameter vector) of the lowest cost in any start's history."""
    best = (np.inf, None, None)
    for f in glob.glob(ms.OUT + 'optimization_history_multistart_start*.csv'):
        with open(f) as fh:
            next(fh, None)
            for line in fh:
                v = line.strip().split(',')
                try:
                    c = float(v[-1])
                    p = np.array(v[1:-1], dtype=float)
                except ValueError:
                    continue  # line still being written
                if c < best[0] and len(p) > 0:
                    best = (c, int(re.search(r'_start(\d+)\.csv$', f).group(1)), p)
    return best


def plate_counts(param, setup, exp_index, t):
    """Modelled plate counts (n_media, n_times) of one experiment."""
    n_cl, n_media = setup['n_cl'], setup['n_media']
    n_exp = len(setup['exps'])
    x0 = 10**param[n_cl*exp_index:n_cl*(exp_index + 1)]
    ode = param[n_cl*n_exp:-n_cl*n_media]
    s_x = param[-n_cl*n_media:].reshape(n_media, n_cl)
    const = [[setup['exp_temps'][setup['exps'][exp_index]]], n_cl, setup['media']]
    C0 = np.concatenate((x0, np.ones(n_cl + 1)))
    C = fm.mdl.model_ODE_solution(setup['model'], t, ode, C0, const)
    n_C = fm.mdl.get_bacterial_count(C, n_cl, 2)
    n_C0 = fm.mdl.get_bacterial_count(C0.reshape(len(C0), 1), n_cl, 2)
    return fm.mdl.observable_MiBi(t, n_C, s_x, n_C0, const, t)[1]


def overview(setup, best, earlier, title, fname):
    """Small multiples of the plate counts, sorted by temperature."""
    days, (mibi, _, _) = setup['data_array']
    days = np.asarray(days, dtype=float)
    exps = setup['exps']
    order = sorted(range(len(exps)), key=lambda i: (setup['exp_temps'][exps[i]], exps[i]))
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.6), sharex=True, sharey=True)
    sq = {'best': [], 'earlier': []}
    for ax, i in zip(axes.ravel(), order):
        d = days[np.any(np.isfinite(mibi[i]), axis=0)]
        t = np.linspace(0., d.max() + 0.5, 150)
        m_best, m_earlier = plate_counts(best, setup, i, t), plate_counts(earlier, setup, i, t)
        for j, med in enumerate(setup['media']):
            col = MEDIUM_COLOR[med]
            ax.plot(t, np.log10(np.maximum(m_earlier[j], 1.)), color=col, lw=1.5, ls=(0, (3, 2)), alpha=.75)
            ax.plot(t, np.log10(np.maximum(m_best[j], 1.)), color=col, lw=2)
            ok = np.isfinite(mibi[i, j]) & (mibi[i, j] > 0)
            ax.plot(days[ok], np.log10(mibi[i, j][ok]), 'o', ms=5, mfc=col, mec='white', mew=.8, zorder=3)
            for key, m in (('best', m_best), ('earlier', m_earlier)):
                sq[key].append(np.interp(days[ok], t, np.log10(np.maximum(m[j], 1.))) - np.log10(mibi[i, j][ok]))
        ax.set_title(f"{exps[i]}, {setup['exp_temps'][exps[i]]:g} °C", fontsize=10, color=TEXT, loc='left')
        ax.grid(color=GRID, lw=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    rmse = {k: float(np.sqrt(np.mean(np.concatenate(v)**2))) for k, v in sq.items()}
    for ax in axes[-1]:
        ax.set_xlabel('Day', color=MUTED, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('log10 count', color=MUTED, fontsize=10)
    handles = [mlines.Line2D([], [], color=MEDIUM_COLOR[m], marker='o', mec='white', ls='', ms=6, label=f'{m}, data')
               for m in ('PC', 'MRS')]
    handles += [mlines.Line2D([], [], color=MUTED, lw=2, label=f"Best fit so far (log10 RMSE {rmse['best']:.2f})"),
                mlines.Line2D([], [], color=MUTED, lw=1.5, ls=(0, (3, 2)),
                              label=f"Earlier parameter set (log10 RMSE {rmse['earlier']:.2f})")]
    fig.legend(handles=handles, loc='upper right', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.99, .955))
    fig.suptitle(title, x=.012, y=.985, ha='left', fontsize=12.5, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    return rmse


def plot_mibi_together(param, setup, dfs, out):
    """Plate counts of all experiments, one panel per temperature and one figure per medium, as in
    plot_results/plot_model_results.py."""
    def measurements(ax0, temp, df, clrs=None, media=''):
        # plot_all passes clrs, plot_measurements_ZL2030_consttemp has its own colors per experiment
        fm.plotting.plot_measurements_ZL2030_consttemp(ax0, temp, df, media=media)

    n_cl, n_media = setup['n_cl'], setup['n_media']
    s_x = param[-n_cl*n_media:].reshape(n_media, n_cl)
    t_model = np.linspace(0., 17., 100)
    _, obs_mibi, _, _, temps = fm.mdl.calc_obs_model(dfs, param[:-n_cl*n_media], dict(setup, s_x=s_x), t_model)
    for j, med in enumerate(setup['media']):
        fm.plotting.plot_all([2., 10., 14.], ('Tag', r'log CFU mL$^{-1}$'), templ_meas=measurements,
                             df=dfs[0].filter(like=med), clrs=plotting_templates.exp_clrs, time_lim=[18., 10.3, 10.3], temps=temps, mtimes=t_model,
                             mestim=obs_mibi[:, j, :], dir=out, add_name=f'MiBi_{med}_const_model')


if __name__ == '__main__':
    os.chdir(ms.MDFM_ROOT)
    out = ms.OUT + 'best_so_far/'
    for sub in ('best/', 'earlier/'):
        os.makedirs(out + sub, exist_ok=True)
    cost, start, param = best_so_far()
    if start is None:
        raise SystemExit('No start has written a history yet.')

    temps = fm.data.EXP_TEMPS
    dfs, _, T_x, s_x_predefined = fm.data.prepare_ZL2030_data(ms.EXPS, ms.EXPS_MALDI, ms.EXPS_NGS, cutoff=ms.CUTOFF,
                                                              cutoff_prop=ms.CUTOFF_PROP, path=out, add_name='_zl2030')
    bact, media = list(dfs[1].index), sorted(set(s.split('_')[-1].split('-')[0] for s in dfs[1].columns))
    n_cl = len(bact)
    setup = fm.pest.define_calibr_setup_insilico(
        {'model': fm.mdl.fusion_model2, 'T_x': T_x, 'workers': 1, 'output_path': out, 'n_cl': n_cl, 'n_media': ms.N_MEDIA,
         'dfs': dfs, 'exps': ms.EXPS, 'exp_temps': {e: temps[e] for e in ms.EXPS}, 'media': media},
        inhib=True, s_x_predefined=s_x_predefined, s_x=None, lambda_exp_max=ms.LAMBDA_EXP_MAX)
    setup['data_array'] = fm.dtf.extract_observables_from_df(dfs)
    earlier = np.array(fm.output.read_from_json(os.path.basename(ms.EARLIER), dir=os.path.dirname(ms.EARLIER) + '/')['param_ode'],
                       dtype=float)

    rmse = overview(setup, param, earlier, f'Plate counts: best fit so far (start {start}, cost {cost:.1f}, not converged) '
                    'vs. earlier parameter set', out + 'overview_plate_counts_best_vs_earlier.png')
    print(f'Best so far: start {start}, cost {cost:.2f}; plate count log10 RMSE {rmse}')

    plot_mibi_together(param, setup, dfs, out)

    colors = dict(zip(bact, fm.plotting.colors_ngs[1:]))
    for sub, p in (('best/', param), ('earlier/', earlier)):
        s_x = p[-n_cl*ms.N_MEDIA:].reshape(ms.N_MEDIA, n_cl)
        fm.plotting.plot_optimization_result(p[:-n_cl*ms.N_MEDIA], dict(setup, s_x=s_x), np.linspace(0, 17, 100),
                                             path=out + sub, clrs=colors, add_name='_zl2030')
    print('Saved to', out)

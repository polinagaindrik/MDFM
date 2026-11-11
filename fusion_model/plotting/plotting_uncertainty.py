"""
Figures of prediction uncertainty (results of fm.pred_unc.predict_ensemble).

plot_prediction_uncertainty(prediction, setup, ...) writes per experiment, like plot_one_exp_model:
    <path>Pred_uncert_<exp><add_name>_realx.pdf        bacterial counts x(t)
    <path>Pred_uncert_<exp><add_name>_ngs.pdf          NGS proportions
    <path>Pred_uncert_<exp><add_name>_<medium>_maldi.pdf  MALDI proportions per medium
    <path>Pred_uncert_<exp><add_name>_platecount.pdf   plate counts per medium
Shaded: band over the ensemble members; line (optional): the estimate; points (optional): the data.
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

from . import plotting_templates as plt_templ
from .plotting_results import medium_label, species_label, species_legend
from ..analysis.prediction_uncertainty import prediction_bands


def _species_color(k, bact, clrs):
    return clrs[bact] if clrs is not None and bact in clrs else plt_templ.colors_ngs[4*k]


def _medium_color(j):
    return plt_templ.colors_ngs[2*j]


def _data_points(df, exp, med=None, row=None):
    """(days, values (n_days, ...)) of one experiment [and medium] from a data frame; None if no data."""
    if df is None:
        return None
    d = df.filter(like=exp + '_')
    if med is not None:
        d = d.filter(like=med)
    if d.shape[1] == 0:
        return None
    days = [float(c.split('_')[3]) for c in d.columns]
    vals = [np.asarray(d[c].loc[row] if row is not None else d[c], dtype=float) for c in d.columns]
    days, vals = zip(*sorted(zip(days, vals), key=lambda z: z[0]))
    return np.array(days), np.array(vals)


def _band_and_estimate(ax, t, lo, hi, est, clr, alpha, plot_estimate, members=None, plot_members=False, noise=None):
    if noise is not None:  # prediction interval of new data (lighter, behind)
        ax.fill_between(t, noise[0], noise[1], color=clr, alpha=0.4 * alpha, linewidth=0)
    ax.fill_between(t, lo, hi, color=clr, alpha=alpha, linewidth=0)
    if plot_members and members is not None:
        for m in members:
            ax.plot(t, m, color=clr, lw=0.5, alpha=0.3)
    if plot_estimate and est is not None:
        ax.plot(t, est, color=clr)


def _legend_handles_style(plot_estimate, plot_data, alpha, conf_label, noise_label=None):
    """Generic handles (gray) explaining band / estimate / data."""
    h = [mpatches.Patch(color='gray', alpha=alpha, label=conf_label)]
    if noise_label:
        h.append(mpatches.Patch(color='gray', alpha=0.4 * alpha, label=noise_label))
    if plot_estimate:
        h.append(mlines.Line2D([], [], color='gray', label='estimate'))
    if plot_data:
        h.append(mlines.Line2D([], [], color='gray', marker='o', ls='', label='data'))
    return h


def _finish(fig, ax, title, fname, legend_handles, style_handles, legend_outside):
    species_legend(ax, legend_handles + style_handles, legend_outside, loc='best', fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_uncertainty(prediction, setup, path='', add_name='', exps=None, bands=None, method=None,
                                quantiles=(0.025, 0.975), plot_estimate=True, plot_data=True, plot_members=False,
                                observables=('x', 'ngs', 'maldi', 'platecount'), clrs=None, alpha=0.25,
                                time_label=r'Time, $t$', legend_outside=False, species=None, file_prefix='Pred_uncert_',
                                x_data=None, noise_bands=None, noise_label=None):
    """
    Plot the prediction band of each experiment (and observable), optionally with the estimate and the data.

    prediction   : result of fm.pred_unc.predict_ensemble
    setup        : setup with 'dfs' (data, also species names), 'exp_temps'
    bands        : precomputed fm.pred_unc.prediction_bands(...); None: computed with method / quantiles
    plot_estimate: line of the estimate (needs predict_ensemble(..., include_estimate=True))
    plot_data    : data points of the experiment from setup['dfs'] (if the experiment is in the data)
    plot_members : thin line of every ensemble member
    observables  : subset of ('x', 'ngs', 'maldi', 'platecount')
    species      : names of the species; None: rows of the MALDI data frame
    x_data       : data frame of x(t) (e.g. the true counts of in-silico data, dataframe_x_*.pkl), shown as
                   points in the x(t) figure; None: setup['dfs'][3] if present
    noise_bands  : fm.pred_unc.predictive_bands(...): prediction interval of new data (parameters + measurement
                   noise), drawn as a lighter band behind the band of the curves (not for x(t), it is not measured)
    """
    method = method or prediction.get('band_method', 'envelope')
    bands = bands if bands is not None else prediction_bands(prediction, method=method, quantiles=quantiles)
    dfs = setup.get('dfs')
    df_platecount, df_maldi, df_ngs = (dfs[0], dfs[1], dfs[2]) if dfs is not None else (None, None, None)
    df_x = x_data if x_data is not None else (dfs[3] if dfs is not None and len(dfs) == 4 else None)
    if species is None:
        species = list(df_maldi.T.columns) if df_maldi is not None else [f'{k:02d}' for k in
                                                                          range(setup['n_cl'])]
    media = prediction['media']
    t = prediction['t']
    cl = prediction.get('confidence_level')
    if method == 'quantile':
        conf_label = rf"{100*(quantiles[1]-quantiles[0]):g}\% credible"
    else:
        conf_label = rf"{100*cl:g}\% confidence" if cl is not None else 'band'
    if noise_bands is not None and noise_label is None:
        noise_label = rf"{100*(quantiles[1]-quantiles[0]):g}\% new data"
    if noise_bands is None:
        noise_label = None

    def nb(exp, o, *idx):
        if noise_bands is None:
            return None
        lo, hi = noise_bands[exp][o]
        return lo[idx], hi[idx]

    estimate = prediction.get('estimate') if plot_estimate else None
    plot_estimate = estimate is not None

    for exp in (exps or prediction['exps']):
        b = bands[exp]
        est = estimate[exp] if estimate is not None else None
        mem = prediction['members'][exp]
        temp = setup['exp_temps'][exp]
        where = f'{exp}, {temp:g}$^\\circ$C'
        base = path + file_prefix + exp + add_name

        if 'x' in observables:
            fig, ax = plt.subplots()
            handles = []
            data = _data_points(df_x, exp) if plot_data else None
            for k, bact in enumerate(species):
                clr = _species_color(k, bact, clrs)
                _band_and_estimate(ax, t, b['x'][0][k], b['x'][1][k], None if est is None else est['x'][k], clr,
                                   alpha, plot_estimate, mem['x'][:, k], plot_members)
                if data is not None:
                    ax.plot(data[0], data[1][:, k], 'o', color=clr)
                handles.append(mlines.Line2D([], [], color=clr, label=species_label(bact)))
            ax.set_yscale('log')
            ax.set_xlim(t[0], t[-1])
            plt_templ.set_labels(fig, ax, time_label, r'log $x(t)$')
            _finish(fig, ax, f'$x(t)$, {where}', base + '_realx.pdf', handles,
                    _legend_handles_style(plot_estimate, data is not None, alpha, conf_label), legend_outside)

        if 'ngs' in observables:
            fig, ax = plt.subplots()
            handles = []
            data = _data_points(df_ngs, exp) if plot_data else None
            for k, bact in enumerate(species):
                clr = _species_color(k, bact, clrs)
                _band_and_estimate(ax, t, b['ngs'][0][k], b['ngs'][1][k], None if est is None else est['ngs'][k],
                                   clr, alpha, plot_estimate, mem['ngs'][:, k], plot_members, nb(exp, 'ngs', k))
                if data is not None:
                    ax.plot(data[0], data[1][:, k], 'o', color=clr)
                handles.append(mlines.Line2D([], [], color=clr, label=species_label(bact)))
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(t[0], t[-1])
            plt_templ.set_labels(fig, ax, time_label, r'$T \mathbf{x} / \lVert T \mathbf{x} \rVert$')
            _finish(fig, ax, f'NGS, {where}', base + '_ngs.pdf', handles,
                    _legend_handles_style(plot_estimate, data is not None, alpha, conf_label, noise_label), legend_outside)

        if 'maldi' in observables:
            for j, med in enumerate(media):
                fig, ax = plt.subplots()
                handles = []
                data = _data_points(df_maldi, exp, med) if plot_data else None
                for k, bact in enumerate(species):
                    clr = _species_color(k, bact, clrs)
                    _band_and_estimate(ax, t, b['maldi'][0][j, k], b['maldi'][1][j, k],
                                       None if est is None else est['maldi'][j, k], clr, alpha, plot_estimate,
                                       mem['maldi'][:, j, k], plot_members, nb(exp, 'maldi', j, k))
                    if data is not None:
                        ax.plot(data[0], data[1][:, k], 'o', color=clr)
                    handles.append(mlines.Line2D([], [], color=clr, label=species_label(bact)))
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(t[0], t[-1])
                plt_templ.set_labels(fig, ax, time_label, r'$S \mathbf{x} / \lVert S \mathbf{x}\rVert$')
                _finish(fig, ax, f'MALDI, {medium_label(med)}, {where}', base + f'_{med}_maldi.pdf', handles,
                        _legend_handles_style(plot_estimate, data is not None, alpha, conf_label, noise_label),
                        legend_outside)

        if 'platecount' in observables:
            fig, ax = plt.subplots()
            handles = []
            any_data = False
            for j, med in enumerate(media):
                clr = _medium_color(j)
                _band_and_estimate(ax, t, b['platecount'][0][j], b['platecount'][1][j],
                                   None if est is None else est['platecount'][j], clr, alpha, plot_estimate,
                                   mem['platecount'][:, j], plot_members, nb(exp, 'platecount', j))
                data = _data_points(df_platecount, exp, med, row='Average') if plot_data else None
                if data is not None:
                    ax.plot(data[0], data[1], 'o', color=clr)
                    any_data = True
                handles.append(mlines.Line2D([], [], color=clr, label=medium_label(med)))
            ax.set_yscale('log')
            ax.set_xlim(t[0], t[-1])
            plt_templ.set_labels(fig, ax, time_label, r'log CFU mL$^{-1}$')
            _finish(fig, ax, f'Plate counts, {where}', base + '_platecount.pdf', handles,
                    _legend_handles_style(plot_estimate, any_data, alpha, conf_label, noise_label), legend_outside)

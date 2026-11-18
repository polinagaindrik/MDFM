import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

from . import plotting_templates as plt_templ
from ..parameter_estimation.media_matrix import Scost_ngsterm, Scost_mediarelation
import fusion_model.tools.dataframe_functions as dtf
import fusion_model.model as mdl

# Set common plotting parameters for all figures
plt.rc('text', usetex=True)
rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

rcParams['lines.linewidth'] = 2.
rcParams['lines.linestyle'] = 'solid' #'dashed'#
rcParams['lines.markersize'] = 8
rcParams['figure.figsize'] = (7.2, 4.5)
rcParams['legend.framealpha'] = 0.
rcParams['legend.handlelength'] = 2.

rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13
rcParams['axes.labelsize'] = 15
rcParams['legend.fontsize'] = 13

rcParams['figure.dpi'] = 500
#rcParams['savefig.format'] = 'pdf'


def define_bacteria_colors(df_ngs, df_maldi):
    """Assign consistent colors to bacteria present in the NGS and MALDI data."""
    df_ngs, bact_ngs = dtf.preprocess_dataframe(df_ngs, cutoff=0.)
    df_maldi, bact_maldi = dtf.preprocess_dataframe(df_maldi, cutoff=0.)
    bact_all = sorted(set(list(bact_ngs.values) + list(bact_maldi.values)))
    clrs1 = {}
    for k, b in enumerate(bact_all):
        clrs1[b] = plt_templ.species_color(k)
    return clrs1


def plot_filteringS2(dfs, opt_param, T_x, exps, path='', clrs=None):
    """Plot the fitted media-filtering weights for each bacterium and experiment."""
    (df_platecount, df_maldi, df_ngs) = dfs
    media = sorted(list(set([s.split('_')[-1].split('-')[0] for s in df_maldi.columns])))
    days_total, [obs_platecount, obs_maldi, obs_ngs] = dtf.extract_observables_from_df(dfs)
    platecount_max = np.nanmax(obs_platecount, axis=(0, 1))
    for k, exp in enumerate(exps):
        rhs1, lhs1 = Scost_ngsterm(days_total, obs_maldi[k], obs_ngs[k], opt_param, T_x)
        rhs2, lhs2 = Scost_mediarelation(days_total, obs_platecount[k]/platecount_max, obs_maldi[k], opt_param)
        for i, med in enumerate (media):
            fig, ax = plt.subplots(figsize=(5, 5))
            for j, bact in enumerate(df_maldi.T.columns):
                ax.scatter(np.array(days_total)[~np.isnan(rhs1[i, j])], rhs1[i, j][~np.isnan(rhs1[i, j])], color=plt_templ.colors_ngs[j], s=60, label=f'{bact}') # ({med})
                ax.plot(np.array(days_total)[~np.isnan(lhs1[i, j])], lhs1[i, j][~np.isnan(lhs1[i, j])], color=plt_templ.colors_ngs[j])
            #ax.set_title('Cost 1')
            fig, ax = plt_templ.set_labels(fig, ax, 'day', r'$J_1$')
            #ax.set_ylim(-0.01, 0.2)
            ax.legend(fontsize=17, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, 1)) # 
            plt.savefig(path+exp+'_'+med+'_cost1.png', bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        for j, bact in enumerate(df_ngs.T.columns):
            ax.scatter(np.array(days_total)[~np.isnan(rhs2[j])], rhs2[j][~np.isnan(rhs2[j])], color=plt_templ.colors_ngs[j], s=60, label=f'{bact}') # ({med})
            ax.plot(np.array(days_total)[~np.isnan(lhs2[j])], lhs2[j][~np.isnan(lhs2[j])], color=plt_templ.colors_ngs[j])
        fig, ax = plt_templ.set_labels(fig, ax, 'day', r'$J_2$')
        ax.legend(fontsize=17, framealpha=0., handlelength=1.5, bbox_to_anchor=(1, 1)) # 
        plt.savefig(path+exp+'_cost2.png', bbox_inches='tight')
        plt.close(fig)
    

def plot_optimization_result(param_x0, calibr_setup, t_model, true_param=None, true_s_x=None, ensemble=None, **kwargs):
    """Plot model predictions against observations for a calibration result.
    true_param: [x0 (log10) | ODE] of the true model (in-silico data), same layout as param_x0, with media matrix
                true_s_x: drawn dashed.
    ensemble  : list of (param [x0 | ODE], s_x) compatible with the data (e.g. fm.pred_unc.ensemble_from_profile_csv,
                mapped to this layout): shaded band = envelope over the members and the fit."""
    data = calibr_setup['dfs']
    exps = sorted(list(set([s.split('_')[0] for s in data[0].columns])))
    n_cl = np.shape(data[1])[0]
    n0 = n_cl*len(exps)
    param_opt = param_x0[n0:]
    for i, exp in enumerate(exps):
        sl = slice(n_cl*i, n_cl*(i+1))
        true = None if true_param is None else \
            (np.asarray(true_param, dtype=float)[n0:], 10**np.asarray(true_param, dtype=float)[sl], true_s_x)
        members = None if ensemble is None else \
            [(np.asarray(p, dtype=float)[n0:], 10**np.asarray(p, dtype=float)[sl], sx) for p, sx in ensemble]
        plot_one_exp_model(exp, param_opt, 10**np.array(param_x0[sl]), calibr_setup, t_model, media=calibr_setup['media'],
                           true=true, members=members, **kwargs)


def plot_prediction_result(param, x0_dict, prediction_setup, t_model, **kwargs):
    """Plot model predictions using estimated initial values for each experiment."""
    exps = prediction_setup['exps']
    param_opt = param
    L0_arr = [10**np.array(x0_dict[exp]) for exp in exps]
    for i, exp in enumerate(exps):
        plot_one_exp_model(exp, param_opt, L0_arr[i], prediction_setup, t_model, media=prediction_setup['media'], **kwargs)


def plot_prediction_result_x0lambda(param, lambda_dict, x0_dict, prediction_setup, t_model, **kwargs):
    """Plot predictions using experiment-specific initial values and lambda estimates."""
    exps = prediction_setup['exps']
    L0_arr = [10**np.array(x0_dict[exp]) for exp in exps]
    for i, exp in enumerate(exps):
        param_opt = np.concatenate((lambda_dict[exp], param))
        plot_one_exp_model(exp, param_opt, L0_arr[i], prediction_setup, t_model, media=prediction_setup['media'], **kwargs)


def is_general_medium(med):
    """General media: gen0, gen1, ..., PC, media1; all others are selective."""
    return med.startswith('gen') or med in ('PC', 'media1')


def medium_label(med):
    """Name of a medium in the figures, e.g. 'general medium gen1'."""
    return f"{'general' if is_general_medium(med) else 'selective'} medium {med}"


def species_label(bact):
    """Name of a species in the figures, as in the parameter names: 'Bacteria_00' -> 'Species 00'
    (paper style: numbered from 1, 'Bacteria_00' -> 'Species 1')."""
    suffix = bact.split('_')[-1]
    if not suffix.isdigit():
        return bact
    start = plt_templ.paper_style['number_from']
    return f'Species {int(suffix) + start}' if start is not None else f'Species {suffix}'


def species_legend(ax, lines, outside=False, **kwargs):
    """Legend of the species: inside the axes with kwargs, or outside right of the axes, centred vertically
    (none in the paper style: save_legend_strip)."""
    if not plt_templ.paper_style['legend']:
        return None
    if outside:
        return ax.legend(handles=lines, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., ncol=1)
    return ax.legend(handles=lines, **kwargs)


def _errorbar(ax, x, y, **kwargs):
    """Data points with error bars in the current style (paper style: light, thin bars without caps)."""
    cont = ax.errorbar(x, y, **{**plt_templ.errorbar_kwargs(), **kwargs})
    if plt_templ.paper_style['thin_errorbars']:
        for bars in cont[2]:
            bars.set_alpha(0.45)
    return cont


def _model_extras(ax, t, true, band, clr):
    """True model (dashed) and band of the fit (shaded) of one curve; returns their maximum (for the y range)."""
    top = -np.inf
    if band is not None:
        ax.fill_between(t, band[0], band[1], color=clr, alpha=0.22, lw=0, zorder=0.5)
        top = max(top, float(np.nanmax(band[1])))
    if true is not None:
        ax.plot(t, true, color=clr, ls=(0, (3, 2)), lw=0.75*rcParams['lines.linewidth'])
        top = max(top, float(np.nanmax(true)))
    return top


def _sub(arr, *idx):
    """arr[idx] of an optional array / (lo, hi) pair."""
    if arr is None:
        return None
    if isinstance(arr, tuple):
        return tuple(_sub(a, *idx) for a in arr)
    out = arr
    for i in idx:
        out = out[i]
    return out


def _panel_letter(ax, coord, letter):
    """Panel letter inside the axes (none in the paper style: added in LaTeX)."""
    if plt_templ.paper_style['panel_label']:
        ax.text(*coord, letter, fontsize=20, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)


def _title(ax, title):
    """Title of a figure (none in the paper style: the caption says what the panel shows)."""
    if title is not None and plt_templ.paper_style['title']:
        ax.set_title(title, fontsize=13)


def save_legend_strip(bact_all, path, media=(), clrs=None, model_label='fitted model', model_ls='-', fname=None,
                      fontsize=None, true_model=False, band_label=None, maldi_band=False):
    """One legend for all panels of a paper figure, two rows filled column by column:
    data / model, [true model / band, MALDI +-1 SD], general / selective medium (if media), species 1 / 2, 3 / 4, ...
    true_model: entry of the dashed true model; band_label: entry of the shaded band (e.g. '95% confidence');
    maldi_band: entry of the MALDI spread band (paper_style['maldi_band']).
    Written to path + fname (default 'Legend_strip.pdf'); include it above the panels in LaTeX."""
    paper = plt_templ.paper_style['tight']   # paper style: printed size (text width 16 cm), 8 pt
    fontsize = fontsize or (8 if paper else 15)
    msize = 3.6 if paper else 8

    def mk(color, marker, label, size=msize, ls='-'):
        return mlines.Line2D([], [], color=color, marker=marker, markeredgecolor='white', markersize=size, ls=ls,
                             label=label)
    h = [mlines.Line2D([], [], color='0.3', ls='none', marker='o', markersize=msize*0.75, label='data'),
         mlines.Line2D([], [], color='0.3', ls=model_ls, label=model_label)]
    if true_model:
        h.append(mlines.Line2D([], [], color='0.3', ls=(0, (3, 2)), lw=0.75*rcParams['lines.linewidth'],
                               label='true model'))
    if band_label:
        h.append(mpatches.Patch(color='0.3', alpha=0.22, lw=0, label=band_label))
    if maldi_band:
        h.append(mpatches.Patch(color='0.3', alpha=0.10, lw=0, label=r'MALDI $\pm$1 SD'))
    if len(h) % 2:   # keep the pairs of the next groups in columns
        h.append(mlines.Line2D([], [], ls='none', label=' '))
    for med in sorted(media, key=lambda m: not is_general_medium(m)):
        h.append(mk(plt_templ.medium_color(med, list(media)), 'o',
                    f"{'general' if is_general_medium(med) else 'selective'} medium"))
    for k, b in enumerate(bact_all):
        h.append(mk(clrs[b] if clrs is not None else plt_templ.species_color(k), plt_templ.species_marker(k),
                    species_label(b)))
    fig = plt.figure(figsize=(6.3, 0.45) if paper else (14, 1.0))
    fig.legend(handles=h, loc='center', ncol=int(np.ceil(len(h) / 2)), frameon=False, fontsize=fontsize,
               handlelength=2.2, columnspacing=1.2 if paper else 1.6)
    fig.savefig(path + (fname or 'Legend_strip.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_one_exp_model(exp, param_opt, L0, prediction_setup, t_model, media=[''], clrs=None, path='', add_name='',
                       relnoise=0.1, maldi_n=None, platecount_log10_sd=None, ngs_dm=None, time_label=r'Time, $t$',
                       legend_outside=False, true=None, members=None):
    """Plot one experiment's predicted bacterial counts and observation types
    (error bars: relnoise * data of plate counts, MALDI and NGS; maldi_n: MALDI as N identified colonies per plate,
    see plot_opt_res_maldi; platecount_log10_sd: plate count noise on log10 of the counts, see plot_opt_res_platecount;
    ngs_dm: NGS as (reads N, concentration A), see plot_opt_res_ngs; time_label: x-axis label, e.g. r'Time, $t$ [day]';
    legend_outside: legend of the species right of the axes, centred vertically;
    true: (ODE parameters, L0, s_x) of the true model, drawn dashed; members: list of such tuples, shaded band =
    envelope over the members and the fit)."""
    data = prediction_setup['dfs']
    n_cl = np.shape(data[1])[0]

    if len(data) == 3:
        (df_platecount0, df_maldi0, df_ngs0, ) = dtf.filter_dataframe(exp+'_', data)
        df_x0 = df_ngs0
    elif len(data) == 4:
        (df_platecount0, df_maldi0, df_ngs0, df_x0, ) = dtf.filter_dataframe(exp+'_', data)
    #temp = float(df_platecount0.columns[0].split("_")[2][:-1])
    temp = prediction_setup['exp_temps'][exp]
    days = dtf.get_meas_days(df_platecount0, exp)
    t_model = np.linspace(0., np.max(days)+0.7, 100)
    const = [[temp], n_cl, media]

    def observables(ode, L0_, s_x):
        """x(t), NGS, MALDI, plate counts on t_model."""
        # lag cells, growing cells 1, inhibitor loads 0 (pool models), exact R(0) (model.ode_structure.initial_state)
        C0 = mdl.initial_state(prediction_setup['model'], L0_, n_cl, ode, temp)
        C = mdl.model_ODE_solution(prediction_setup['model'], t_model, ode, C0, const)
        nC = mdl.get_bacterial_count(C, n_cl, 2)
        nC0 = mdl.get_bacterial_count(np.array(C0).reshape(len(C0), 1), n_cl, 2)
        _, o_ngs = mdl.observable_NGS(t_model, nC, prediction_setup['T_x'], nC0, const, t_model)
        _, o_maldi = mdl.observable_MALDI(t_model, nC, s_x, nC0, const, t_model)
        _, o_pc = mdl.observable_platecount(t_model, nC, s_x, nC0, const, t_model)
        return dict(x=np.asarray(nC, dtype=float), ngs=np.asarray(o_ngs, dtype=float),
                    maldi=np.asarray(o_maldi, dtype=float), pc=np.asarray(o_pc, dtype=float))

    fit = observables(param_opt, L0, prediction_setup['s_x'])
    n_C, obs_ngs, obs_maldi, obs_platecount = fit['x'], fit['ngs'], fit['maldi'], fit['pc']
    days_ngs = days_maldi = days_platecount = t_model
    obs_true = observables(*true) if true is not None else {}
    band = {}
    if members:
        curves = [observables(*m) for m in members] + [fit]
        band = {k: (np.nanmin([c[k] for c in curves], axis=0), np.nanmax([c[k] for c in curves], axis=0)) for k in fit}

    #f_x = mdl.media_filtering(t_model, n_C, prediction_setup['s_x'], n_C0, const)

    where = f'{exp}, {temp:g}$^\\circ$C'
    # real data has no measurement of x (df_x0 is the NGS data, only for the days and species): model curves only
    plot_opt_res_realx(df_x0, t_model, n_C, exp, path=path+'Param_estim_', clrs=clrs, add_name=add_name,
                       title=f'Cells, {where}' if len(data) == 4 else f'$x(t)$, {where}', time_label=time_label,
                       legend_outside=legend_outside, plot_meas=len(data) == 4, obs_true=obs_true.get('x'),
                       band=band.get('x'))
    #for i in range (len(media)):
    #    plot_opt_res_realx(df_x0, t_model, f_x[i], exp, path=path+f'Sx_{media[i]}_', clrs=clrs, add_name=add_name, plot_meas=False,
    #                       title=f'Cells recovered on {medium_label(media[i])}, {where}')
    plot_opt_res_ngs(df_ngs0, days_ngs, obs_ngs, exp, path=path+'Param_estim_', clrs=clrs, add_name=add_name,
                     relnoise=relnoise, ngs_dm=ngs_dm, title=f'NGS, {where}', time_label=time_label, legend_outside=legend_outside,
                     obs_true=obs_true.get('ngs'), band=band.get('ngs'))
    plot_opt_res_maldi(df_maldi0, days_maldi, obs_maldi, exp, media=media, path=path+'Param_estim_', clrs=clrs,
                       add_name=add_name, relnoise=relnoise, maldi_n=maldi_n, title=where, time_label=time_label, legend_outside=legend_outside,
                       obs_true=obs_true.get('maldi'), band=band.get('maldi'))
    plot_opt_res_platecount(df_platecount0, days_platecount, obs_platecount, exp, media=media, path=path+'Param_estim_', add_name=add_name,
                      relnoise=relnoise, log10_sd=platecount_log10_sd, title=f'Plate counts, {where}', time_label=time_label,
                      obs_true=obs_true.get('pc'), band=band.get('pc'))


def plot_opt_res_ngs(df_ngs0, days_model, obs_model, exp, std=None, path='', add_name='', clrs=None, relnoise=0.1,
                     title=None, ngs_dm=None, relnoise_of_model=False, time_label=r'Time, $t$', legend_outside=False,
                     obs_true=None, band=None):
    """Plot modeled NGS proportions against measurements for one experiment
    (error bars std, if None relnoise * data; ngs_dm = (N, A): Dirichlet-multinomial standard deviation;
    relnoise_of_model: relnoise * model at the measured days, the SD of data with relative Gaussian noise)."""
    fig, ax = plt.subplots()
    fig.subplots_adjust()
    days_meas = [float(f.split('_')[3]) for f in df_ngs0.columns]
    obs_meas = np.array([df_ngs0[f] for f in df_ngs0.columns])
    days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))
    lines, tops = [], []
    for k, (o_n, o_meas, bact) in enumerate(zip(np.array(obs_model), np.array(obs_meas).T, df_ngs0.T.columns)):
        err = relnoise*o_meas if std is None else std
        if std is None and ngs_dm is not None:
            err = np.sqrt(o_meas * (1 - o_meas) * (ngs_dm[0] + ngs_dm[1]) / (ngs_dm[0] * (1 + ngs_dm[1])))
        elif std is None and relnoise_of_model:
            err = relnoise*np.interp(days_meas, days_model, o_n)
        if clrs is not None:
            clr = clrs[bact]
        else:
            clr = plt_templ.species_color(k)
        mrk = plt_templ.species_marker(k)
        _errorbar(ax, plt_templ.dodge(days_meas, k, len(obs_model)), o_meas, yerr=err, fmt=mrk, color=clr, label=f'{bact}')
        tops.append(max(plt_templ.errorbar_top(o_meas, err), float(np.nanmax(o_n)),
                        _model_extras(ax, days_model, _sub(obs_true, k), _sub(band, k), clr)))
        #ax.scatter(days_model, o_n,label=f'{bact}', color='w', marker='o', s=40, edgecolors=clr)
        ax.plot(days_model, o_n, color=clr)
        lines.append(mlines.Line2D([], [], color=clr, marker=mrk, label=species_label(bact)))
    fig, ax = plt_templ.set_labels(fig, ax, time_label, r'$T \mathbf{x   } / \lVert T \mathbf{x} \rVert$')
    #fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'$T \mathbf{x   } / \lVert T \mathbf{x} \rVert$')
    #ax.set_ylim(-0.05, 1.05)
    if plt_templ.paper_style['tight']:   # no headroom for a legend
        top = max(tops)
        ax.set_ylim(-0.025*top, top*1.04)
        plt_templ.tight_xlim(ax, days_meas, len(obs_model))
    else:
        ax.set_ylim(-0.05, np.max(obs_meas)*(1+0.2)+0.1)
        ax.set_xlim(np.min(days_meas)-0.25, np.max(days_meas)+0.5)
    coord_text = (0.06, 0.92)
    _panel_letter(ax, coord_text, '(b)')
    species_legend(ax, lines, legend_outside, ncol=2)
    _title(ax, title)
    #ax.legend(fontsize=13, framealpha=0., handlelength=2., bbox_to_anchor=(1.,1.))# bbox_to_anchor=(1., 0.1, 0.5, 0.5))#, bbox_to_anchor=(1, 1)) # 
    plt.savefig(path+exp+add_name+'_ngs.pdf', bbox_inches='tight')
    plt.close(fig)


def wilson_error_bars(p, n, z=1.):
    """Lower and upper error bar (y - low, high - y) of proportions p from n identified colonies: Wilson score
    interval (z = 1: about +- 1 SD); unlike sqrt(p(1-p)/n) it is also informative for p = 0 and p = 1."""
    p = np.asarray(p, dtype=float)
    center = (p + z**2/(2*n)) / (1 + z**2/n)
    half = z/(1 + z**2/n) * np.sqrt(p*(1 - p)/n + z**2/(4*n**2))
    return np.array([np.maximum(p - (center - half), 0.), np.maximum(center + half - p, 0.)])


def plot_opt_res_maldi(df_maldi0, days_model, obs_model, exp, std=None, media=[' '], path='', add_name='', clrs=None,
                       relnoise=0.1, maldi_n=None, title=None, relnoise_of_model=False, time_label=r'Time, $t$', legend_outside=False,
                       obs_true=None, band=None):
    """Plot modeled MALDI proportions against measurements for one experiment
    (error bars std, if None relnoise * data; maldi_n = N identified colonies per plate: wilson_error_bars;
    relnoise_of_model: relnoise * model at the measured days, the SD of data with relative Gaussian noise).
    title: figure title after 'MALDI, <medium>', each medium has its own figure and panel letter."""
    general = [is_general_medium(m) for m in media]
    for j, med in enumerate(media):
        fig, ax = plt.subplots()
        fig.subplots_adjust()
        days_meas = [float(f.split('_')[3]) for f in df_maldi0.filter(like=med).columns]
        obs_meas = np.array([df_maldi0.filter(like=med)[f] for f in df_maldi0.filter(like=med).columns])
        days_meas, obs_meas = zip(*sorted(zip(days_meas, obs_meas)))
        o_model = obs_model[j]
        lines, tops = [], []
        for k, (o_m, o_meas, bact) in enumerate(zip(np.array(o_model), np.array(obs_meas).T, df_maldi0.T.columns)):
            if clrs is not None:
                clr = clrs[bact]
            else:
                clr = plt_templ.species_color(k)
            mrk = plt_templ.species_marker(k)
            if std is not None:
                err = std
            elif maldi_n is not None:
                err = wilson_error_bars(o_meas, maldi_n)
            elif relnoise_of_model:
                err = relnoise*np.interp(days_meas, days_model, o_m)
            else:
                err = relnoise*o_meas
            if plt_templ.paper_style['maldi_band'] and maldi_n is not None:
                # no error bars: expected spread of the data around the model (+-1 SD of the share, maldi_n colonies)
                sd = np.sqrt(np.clip(o_m*(1 - o_m), 0., None)/maldi_n)
                ax.fill_between(days_model, np.clip(o_m - sd, 0., None), o_m + sd, color=clr, alpha=0.10, lw=0, zorder=0.4)
                kw = plt_templ.errorbar_kwargs()
                ax.plot(plt_templ.dodge(days_meas, k, len(o_model)), o_meas, mrk, color=clr, ls='none', label=f'{bact}',
                        **{key: kw[key] for key in ('markeredgecolor', 'markeredgewidth', 'markersize') if key in kw})
                tops.append(max(float(np.nanmax(o_meas)), float(np.nanmax(o_m + sd))))
            else:
                _errorbar(ax, plt_templ.dodge(days_meas, k, len(o_model)), o_meas, yerr=err, fmt=mrk, color=clr, label=f'{bact}')
                tops.append(max(plt_templ.errorbar_top(o_meas, err), float(np.nanmax(o_m))))
            tops.append(_model_extras(ax, days_model, _sub(obs_true, j, k), _sub(band, j, k), clr))
            ax.plot(days_model, o_m, color=clr)
            lines.append(mlines.Line2D([], [], color=clr, marker=mrk, label=species_label(bact)))
        if plt_templ.paper_style['tight']:   # no headroom for a legend
            top = np.nanmax(tops)
            ax.set_ylim(-0.025*top, top*1.04)
            plt_templ.tight_xlim(ax, days_meas, len(o_model))
        else:
            ax.set_ylim(-0.05, np.nanmax([np.nanmax(obs_meas), np.nanmax(o_model)])*1.1)#+0.1)
            ax.set_xlim(np.min(days_meas)-0.25, np.max(days_meas)+0.5)
        fig, ax = plt_templ.set_labels(fig, ax, time_label, r'$S \mathbf{x} / \lVert S \mathbf{x}\rVert$')
        #fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'$S \mathbf{x} / \lVert S \mathbf{x}\rVert$')
        
        coord_text = (0.06, 0.92)
        # (c) general, (d) selective medium; two media of the same kind: in their order
        first = j == 0 if len(media) > 1 and len(set(general)) == 1 else general[j]
        _panel_letter(ax, coord_text, '(c)' if first else '(d)')
        if is_general_medium(med):
            species_legend(ax, lines, legend_outside, bbox_to_anchor=(0.72, 0.3))
        else:
            species_legend(ax, lines, legend_outside)
        _title(ax, None if title is None else f'MALDI, {medium_label(med)}, {title}')
        #ax.legend(bbox_to_anchor=(1., 1.))#, bbox_to_anchor=(1., 0.1, 0.5, 0.5)) #, bbox_to_anchor=(1, 1)
        plt.savefig(path+exp+add_name+f'_{med}'+'_maldi.pdf', bbox_inches='tight')
        plt.close(fig)
    

def plot_opt_res_platecount(df_platecount0, days_model, obs_model, exp, std=None, media=[''], path='', add_name='', clrs=None,
                      relnoise=0.1, log10_sd=None, title=None, relnoise_of_model=False, time_label=r'Time, $t$',
                      obs_true=None, band=None):
    """Plot modeled plate counts counts against measurements for one experiment (error bars relnoise * data;
    log10_sd: noise on log10 of the counts, error bars +- log10_sd in log10 units;
    relnoise_of_model: relnoise * model at the measured days, the SD of data with relative Gaussian noise)."""
    fig, ax = plt.subplots()
    fig.subplots_adjust()
    lines = []
    for j, med in enumerate(media):
        med_label = medium_label(med)
        days_meas = [float(f.split('_')[3]) for f in df_platecount0.filter(like=med).columns]
        obs_meas = np.array([df_platecount0.filter(like=med)[f]['Average'] for f in df_platecount0.filter(like=med).columns])
        stds = np.array([df_platecount0.filter(like=med)[f]['Standard deviation'] for f in df_platecount0.filter(like=med).columns])
        stds = relnoise*obs_meas
        days_meas, obs_meas, stds = zip(*sorted(zip(days_meas, obs_meas, stds)))
        if log10_sd is not None:
            stds = np.array([np.array(obs_meas)*(1 - 10.**(-log10_sd)), np.array(obs_meas)*(10.**log10_sd - 1)])
        o_model = obs_model[j]
        if log10_sd is None and relnoise_of_model:
            stds = relnoise*np.interp(days_meas, days_model, o_model)
        med_clr = plt_templ.medium_color(med, media)
        ax.plot(days_model, o_model, color=med_clr, label=med_label+' (model)')
        _errorbar(ax, days_meas, obs_meas, yerr=stds, fmt='o', color=med_clr, label=med_label+' (data)')
        lines.append(mlines.Line2D([], [], color=med_clr, marker='o', label=med_label))
        _model_extras(ax, days_model, _sub(obs_true, j), _sub(band, j), med_clr)
    coord_text = (0.06, 0.92)
    _panel_letter(ax, coord_text, '(a)')
    ax.set_yscale('log')
    if plt_templ.paper_style['tight']:
        plt_templ.tight_xlim(ax, days_meas, 1)
    else:
        ax.set_xlim(np.min(days_meas)-0.25, np.max(days_meas)+0.5)
    fig, ax = plt_templ.set_labels(fig, ax, time_label, plt_templ.paper_style['ylabel_pc'])
    #fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'log CFU mL$^{-1}$')
    if plt_templ.paper_style['legend']:
        ax.legend()#, handles=lines)
    _title(ax, title)
    plt.savefig(path+exp+add_name+'_platecount.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_opt_res_realx(df0, days_model, obs_model, exp, path='', add_name='', clrs=None, plot_meas=True, title=None, time_label=r'Time, $t$', legend_outside=False,
                       obs_true=None, band=None):
    """Plot modeled bacterial trajectories against measured count data.
    title: figure of a fit result, full time range and y-range of the curves, legend inside the axes, no panel
    letter; None: layout of the data-generation figures (panel letter = temperature, days 0-10 above 2 C)."""
    fig, ax = plt.subplots()
    fig.subplots_adjust()
    bact_all = df0.T.columns
    days_meas = [float(f.split('_')[3]) for f in df0.columns]
    obs_meas = np.array([df0[f] for f in df0.columns])
    temp = float(df0.columns[0].split("_")[2][:-1])
    lines = []
    for k, (o_n, o_meas) in enumerate(zip(np.array(obs_model), np.array(obs_meas).T)):
        if clrs is not None:
            clr = clrs[bact_all[k]]
        else:
            clr = plt_templ.species_color(k)
        mrk = plt_templ.species_marker(k)
        ax.plot(days_model, o_n, label=f'{bact_all[k]}', color=clr)
        _model_extras(ax, days_model, _sub(obs_true, k), _sub(band, k), clr)
        if plot_meas:
            _errorbar(ax, plt_templ.dodge(days_meas, k, len(bact_all)), o_meas, yerr=0., fmt=mrk, color=clr,
                        label=f'{bact_all[k]}')
        lines.append(mlines.Line2D([], [], color=clr, marker=mrk if plot_meas else '', label=species_label(bact_all[k])))
    ax.set_yscale('log')
    if title is not None:
        shown = np.array(obs_model, dtype=float)[:, np.asarray(days_model) <= np.max(days_meas)]
        low, high = (np.min(obs_meas), np.max(obs_meas)) if plot_meas else (np.min(shown), np.max(shown))
        if plt_templ.paper_style['tight']:   # no headroom for a legend
            plt_templ.tight_xlim(ax, days_meas, len(bact_all) if plot_meas else 1)
            ax.set_ylim(low*0.7, high*1.6)
        else:
            ax.set_xlim(0., np.max(days_meas)+0.3)
            ax.set_ylim(low*0.5, high*2.5)
        fig, ax = plt_templ.set_labels(fig, ax, time_label, plt_templ.paper_style['ylabel_x'])
        species_legend(ax, lines, legend_outside, ncol=max(1, len(bact_all)//4))
        if title:
            _title(ax, title)
        plt.savefig(path+exp+add_name+'_realx.pdf', bbox_inches='tight')
        plt.close(fig)
        return
    #ax.set_xlim(-0.25, np.max(days_meas)+0.3)
    ax.set_xlim(0., np.max(days_meas)+0.3)
    ax.set_ylim(np.min(obs_meas)*0.5, np.max(obs_meas)*2.5)
    if temp != 2.0:
        #ax.set_xlim(-0.25, 10)
        ax.set_xlim(0., 10)
        ax.set_ylim(np.min(obs_meas)*0.6, np.max(obs_meas)*2.7)
    if plt_templ.paper_style['tight']:   # no headroom for a legend
        ax.set_ylim(np.min(obs_meas)*0.7, np.max(obs_meas)*1.6)
    fig, ax = plt_templ.set_labels(fig, ax, time_label, plt_templ.paper_style['ylabel_x'])
    #fig, ax = plt_templ.set_labels(fig, ax, 'Tag', r'log CFU mL$^{-1}$')
    #coord_text = (0.95, 0.07)
    coord_text = (0.06, 0.92)
    if temp == 2.0:
        text_lbl = '(a)'
        legend_box = [0.2, 0.5]
    elif temp == 6.0:
        text_lbl = '(b)'
        legend_box = [0.2, 0.5]
    else:
        text_lbl = '(c)'
        legend_box = [0.317, 0.427]
    _panel_letter(ax, coord_text, text_lbl)
    if plt_templ.paper_style['legend']:
        ax.legend(ncol=max(1, len(bact_all)//4), handles=lines, bbox_to_anchor=legend_box, bbox_transform=fig.transFigure)
    #ax.legend(handlelength=1.3, ncol=len(bact_all)//4)#, bbox_to_anchor=(1, 1))
    plt.savefig(path+exp+add_name+'_realx.pdf', bbox_inches='tight')
    plt.close(fig)
"""
Groups of species that share one inhibitor strength, chosen from the data before the fit
========================================================================================

With the shared inhibitor load (calibr_setup['k_structure'] = 'shared') the growth of species i is divided by
1 + kappa a_i I(t), I(t) = sum_j b_j x2_j(t). The data determine the load, not always the individual strengths b_j:

  1) a species that stays rare contributes nothing measurable to I(t), whatever its b_j;
  2) species whose abundance curves are proportional over all experiments and days enter I(t) only through the sum of
     their strengths (collinearity, as in a regression).

Both can be read from the data without the ODE model, from the NGS shares alone (share_curves). Plate counts are not
used: they depend on the selectivity of the medium, which the model describes separately. NGS shares are enough because
x_j(t) = q_j(t) N(t) with the same total N(t) for all species, so two species have proportional abundances exactly when
their shares are proportional. Then (cluster_inhibitors):

  - species whose mean NGS share is below min_share form one 'background' group;
  - the other species are clustered by the cosine similarity of their share curves over all experiments and days
    (average linkage); species with curves more similar than min_similarity share one strength.

The species keep their own states, growth, lag, sensitivity a_i and media coefficients; only b is shared within a group.
The groups say which strengths the data can separate, not which biological mechanism acts; they depend on the data set.

    curves = fm.inhib_groups.share_curves(calibr_setup['data_array'])
    res = fm.inhib_groups.cluster_inhibitors(curves)
    calibr_setup['k_groups'] = res['groups']       # with calibr_setup['k_structure'] = 'shared'
    fm.inhib_groups.plot_inhibition_groups(res, species_names, path + 'inhibition_groups.png')

Model-based, in two steps (model_curves, method 'model'): fit without groups, compute the fitted absolute curves
x2_j(t) of the growing cells (the media are handled by the model through S), cluster them with the same two rules
(background by share of the fitted abundance, groups by proportional curves), then refit with the groups. This measures
what the inhibitor load depends on, the absolute abundance, without combining NGS with medium-selective plate counts.

Alternatively the groups come from taxonomy (taxonomy_groups): a table genus -> group, by default the lactic acid
bacteria (acid and bacteriocin producers in chilled meat) versus all others. inhibition_groups(method=...) chooses:
'none' (every species its own strength), 'abundance' (from the data, as above) or 'taxonomy'.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

MIN_SHARE = 0.01
MIN_SIMILARITY = 0.9
METHODS = ('none', 'abundance', 'model', 'taxonomy')
# default taxonomy grouping: lactic acid bacteria (incl. the genera split from Lactobacillus in 2020) versus all others
LACTIC_ACID_BACTERIA = ('Carnobacterium', 'Lactobacillus', 'Lactococcus', 'Leuconostoc', 'Weissella', 'Pediococcus',
                        'Enterococcus', 'Streptococcus', 'Vagococcus', 'Latilactobacillus', 'Dellaglioa',
                        'Lactiplantibacillus', 'Companilactobacillus', 'Lacticaseibacillus', 'Fructilactobacillus')
DEFAULT_TAXONOMY = {g: 'lactic acid bacteria' for g in LACTIC_ACID_BACTERIA}


def share_curves(data_array):
    """
    NGS share of each species over all experiments and days, (n_species, n_points), from data_array = (days,
    [platecount, maldi, ngs]) as calibr_setup['data_array']; samples without NGS are left out. Only NGS is used: plate
    counts depend on the medium, and proportional shares already mean proportional abundances (see the module docstring).
    """
    _, (_, _, ngs) = data_array
    ngs = np.asarray(ngs, dtype=float)                      # (experiments, species, days)
    x = np.transpose(ngs, (1, 0, 2)).reshape(ngs.shape[1], -1)
    measured = np.isfinite(x).all(axis=0)
    return x[:, measured]


def model_curves(param, calibr_setup, days=None):
    """
    Fitted absolute curves of the growing cells x2_j(t) of every species over all experiments and days,
    (n_species, n_points), from a full parameter vector [x0 | ODE | S] and its calibr_setup (structure, groups and
    lambda form are taken into account). days None: the measurement days of calibr_setup['data_array'].
    """
    from ..model.solving import model_ODE_solution
    from ..parameter_estimation.cost_function import ode_params_for_model
    p = np.asarray(param, dtype=float)
    n, n_media, exps = calibr_setup['n_cl'], calibr_setup['n_media'], calibr_setup['exps']
    days = np.asarray(calibr_setup['data_array'][0] if days is None else days, dtype=float)
    x0 = p[:n * len(exps)].reshape(len(exps), n)
    ode = ode_params_for_model(p[n * len(exps):-n * n_media], calibr_setup)
    curves = []
    for i, e in enumerate(exps):
        c0 = np.concatenate((10. ** x0[i], np.ones(n + 1)))
        c = model_ODE_solution(calibr_setup['model'], days, ode, c0, [[calibr_setup['exp_temps'][e]], n, calibr_setup['media']])
        curves.append(np.asarray(c)[n:2 * n])                       # x2 (species, days)
    return np.concatenate(curves, axis=1)


def cluster_inhibitors(curves, min_share=MIN_SHARE, min_similarity=MIN_SIMILARITY):
    """
    Groups of species sharing one strength (see the module docstring). Returns a dict with
      groups      label per species, for calibr_setup['k_groups']: 0 = background (rare species), 1, 2, ... clusters
      share       mean NGS share of each species over all samples
      similarity  cosine similarity of the curves, (n, n)
      clusters    list of the species indices of every group, background first if present
    """
    V = np.asarray(curves, dtype=float)
    n = V.shape[0]
    share = V.mean(axis=1) / max(V.mean(axis=1).sum(), 1e-300)
    norm = np.linalg.norm(V, axis=1)
    U = V / np.where(norm > 0, norm, 1.)[:, None]
    similarity = U @ U.T
    rare = share < min_share
    groups = np.zeros(n, dtype=int)
    abundant = np.where(~rare)[0]
    if len(abundant) == 1:
        groups[abundant] = 1
    elif len(abundant) > 1:
        Z = linkage(U[abundant], method='average', metric='cosine')
        groups[abundant] = fcluster(Z, t=1. - min_similarity, criterion='distance')
    if not rare.any():
        groups = groups - groups.min()      # no background group: labels from 0
    order = sorted(set(groups), key=lambda g: (g != 0 or not rare.any(), min(np.where(groups == g)[0])))
    relabel = {g: k for k, g in enumerate(order)}
    groups = np.array([relabel[g] for g in groups])
    clusters = [[int(j) for j in np.where(groups == g)[0]] for g in range(len(order))]
    return {'groups': [int(g) for g in groups], 'share': share, 'similarity': similarity, 'clusters': clusters,
            'background': bool(rare.any()), 'min_share': min_share, 'min_similarity': min_similarity}


def taxonomy_groups(names, table=None, default='other'):
    """
    Groups from taxonomy: table maps a genus name (the first word of each species name, e.g. 'Leuconostoc gelidum' ->
    'Leuconostoc') to a group name; names not in the table go to default. table None: DEFAULT_TAXONOMY (lactic acid
    bacteria versus all others). Returns a dict like cluster_inhibitors: groups (labels 0, 1, ... in order of first
    appearance), group_names, clusters.
    """
    table = DEFAULT_TAXONOMY if table is None else table
    genus = [str(nm).replace('_', ' ').split()[0] for nm in names]
    group_names = [table.get(g, default) for g in genus]
    labels = {}
    for g in group_names:
        labels.setdefault(g, len(labels))
    groups = [labels[g] for g in group_names]
    return {'groups': groups, 'group_names': list(labels), 'clusters': [[j for j, gg in enumerate(groups) if gg == k]
                                                                       for k in range(len(labels))]}


def inhibition_groups(method, data_array=None, media=None, names=None, general_medium=None, table=None,
                      min_share=MIN_SHARE, min_similarity=MIN_SIMILARITY, param=None, calibr_setup=None):
    """
    Groups for calibr_setup['k_groups'] by method: 'none' -> None (every species its own strength), 'abundance' ->
    cluster_inhibitors on share_curves (needs data_array only), 'model' -> cluster_inhibitors on model_curves (needs
    param of a fit without groups and its calibr_setup), 'taxonomy' -> taxonomy_groups (needs
    names, optional table). Returns (groups or None, result dict or None).
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, got {method!r}')
    if method == 'none':
        return None, None
    if method == 'taxonomy':
        res = taxonomy_groups(names, table)
        return res['groups'], res
    if method == 'model':
        res = cluster_inhibitors(model_curves(param, calibr_setup), min_share, min_similarity)
        return res['groups'], res
    res = cluster_inhibitors(share_curves(data_array), min_share, min_similarity)
    return res['groups'], res


def group_strengths(b, groups, curves):
    """Strength of each group that gives about the same load as individual strengths b: mean of the members' b weighted
    by their summed curves (sum_j b_j X_j / sum_j X_j; with share curves this weights by the mean share, an approximation
    of the load-equivalent strength). Returns b per species (equal within a group)."""
    b = np.asarray(b, dtype=float)
    X = np.asarray(curves, dtype=float).sum(axis=1)
    out = b.copy()
    for g in set(groups):
        idx = [j for j, gg in enumerate(groups) if gg == g]
        w = X[idx]
        out[idx] = np.sum(b[idx] * w) / w.sum() if w.sum() > 0 else b[idx].mean()
    return out


def plot_inhibition_groups(result, names=None, save_path=None, title=None):
    """Two panels: share of each species in the total abundance (log scale, background threshold) and the cosine
    similarity of the abundance curves, species ordered by group, groups outlined."""
    import matplotlib.pyplot as plt
    with plt.rc_context({'text.usetex': False, 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8}):   # the package sets usetex and large fonts globally
        return _plot_inhibition_groups(result, names, save_path, title)


def _plot_inhibition_groups(result, names, save_path, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    share, sim, groups = result['share'], result['similarity'], np.array(result['groups'])
    n = len(share)
    names = [str(i + 1) for i in range(n)] if names is None else list(names)
    order = [j for c in result['clusters'] for j in c]
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948', '#1c5cab', '#b5651d']
    cmap = lambda g: palette[g % len(palette)]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1, 1.25]})
    colours = ['#b9b9b9' if (result['background'] and groups[j] == 0) else cmap(groups[j]) for j in order]
    ax[0].barh(range(n), [max(share[j], 1e-6) for j in order], color=colours)
    ax[0].set_xscale('log'); ax[0].invert_yaxis()
    ax[0].set_yticks(range(n)); ax[0].set_yticklabels([names[j] for j in order], fontsize=8)
    ax[0].axvline(result['min_share'], color='black', lw=1, ls='--')
    ax[0].text(result['min_share'], -0.7, f" background below {result['min_share']:.0%}", fontsize=7.5, va='bottom')
    ax[0].set_xlabel('mean NGS share'); ax[0].set_title('Relative abundance of each species', fontsize=10)
    im = ax[1].imshow(sim[np.ix_(order, order)], vmin=0, vmax=1, cmap='Blues')
    ax[1].set_xticks(range(n)); ax[1].set_xticklabels([names[j] for j in order], rotation=90, fontsize=8)
    ax[1].set_yticks(range(n)); ax[1].set_yticklabels([names[j] for j in order], fontsize=8)
    pos = 0
    for k, c in enumerate(result['clusters']):
        col = '#7a7a7a' if (result['background'] and k == 0) else cmap(groups[c[0]])
        ax[1].add_patch(Rectangle((pos - 0.5, pos - 0.5), len(c), len(c), fill=False, lw=2, ec=col))
        pos += len(c)
    for i in range(n):
        for j in range(n):
            v = sim[order[i], order[j]]
            ax[1].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white' if v > 0.6 else 'black')
    fig.colorbar(im, ax=ax[1], fraction=0.046, label='cosine similarity of the NGS share curves')
    ax[1].set_title(f"Similarity of the share curves (shared strength above {result['min_similarity']})", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

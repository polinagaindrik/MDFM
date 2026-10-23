import matplotlib.pyplot as plt
import numpy as np
from . import plotting_templates as plt_templ


def lin_func(x, a0, a1):
    """Evaluate a linear function with intercept a0 and slope a1."""
    return a0 + a1*x

def exp_func(x, a0, a1):
    """Evaluate an exponential parameter curve from its coefficients."""
    return a0*x**a1

EST_LABEL = 'estimated'
REAL_LABEL = 'true'
REAL_MARKER = dict(marker='x', color='#D06062', markersize=11, markeredgewidth=2.5, linestyle='none')


def _curve_legends(ax, bact_all, clrs1, has_real):
    """Two legends for temperature curves: species (colour) and estimated/true (line style)."""
    from matplotlib.lines import Line2D
    species = [Line2D([], [], color=clrs1[b], linewidth=2, label=b) for b in bact_all]
    leg1 = ax.legend(handles=species, fontsize=12, framealpha=0.1, bbox_to_anchor=(1, 1), loc='upper left')
    if has_real:
        ax.add_artist(leg1)
        styles = [Line2D([], [], color='k', linewidth=2, label=EST_LABEL),
                  Line2D([], [], color='k', linewidth=2, linestyle='dashed', label=REAL_LABEL)]
        ax.legend(handles=styles, fontsize=12, framealpha=0.1, bbox_to_anchor=(1, 0), loc='lower left')


def _stem_est_real(ax, bact_all, est, real=None):
    """Estimated values as stems, true values as crosses, with a legend."""
    from matplotlib.lines import Line2D
    ax.stem(bact_all, est)
    if real is not None:
        ax.plot(bact_all, real, **REAL_MARKER)
        handles = [Line2D([], [], color='C0', marker='o', label=EST_LABEL),
                   Line2D([], [], label=REAL_LABEL, **REAL_MARKER)]
        ax.legend(handles=handles, fontsize=12, framealpha=0.1, bbox_to_anchor=(1, 1), loc='upper left')


def plot_alpha(alph, bact_all, clrs1, alph_real=None, path=''):
    """Plot estimated alpha growth parameters against temperature for each bacterium."""
    temp = np.linspace(1.95, 14.05, 100)
    (alph_1, alph_exp) = alph
    fig, ax = plt.subplots()
    for a0, a1, b in zip(alph_1, alph_exp, bact_all):
        #ax.plot(temp, lin_func(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2)
        ax.plot(temp, exp_func(temp, a0, a1)/24., color=clrs1[b], linewidth=2)
    if alph_real is not None:
        (alph_1_real, alph_exp_real) = alph_real
        for a0, a1, b in zip(alph_1_real, alph_exp_real, bact_all):
            ax.plot(temp, exp_func(temp, a0, a1)/24., color=clrs1[b], linewidth=2, linestyle='dashed')
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.set_xlabel(r'Temperatur [°C]', fontsize=14)
    ax.set_xlim(1.95, 14.05)
    ax.tick_params(axis='both', which='major', labelsize=12)
    _curve_legends(ax, bact_all, clrs1, alph_real is not None)
    plt.savefig(path+'alphas.png', bbox_inches='tight')
    plt.close(fig)

    # Now plot separately alpha parameter values:
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    _stem_est_real(ax, bact_all, alph_1, None if alph_real is None else alph_1_real)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(path+'alphas_1.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    _stem_est_real(ax, bact_all, alph_exp, None if alph_real is None else alph_exp_real)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(path+'alphas_exp.png', bbox_inches='tight')
    plt.close(fig)


def exp_func2(x, a0, a1):
    """Evaluate the exponential temperature-response function used for lambda."""
    return 10**a0 * x**a1

def plot_lambda(lambd, bact_all, clrs1, lambd_real=None, path=''):
    """Plot estimated lambda growth rates against temperature for each bacterium."""
    n_cl = len(bact_all)
    (lambd_1, lambd_exp) = lambd
    fig, ax = plt.subplots()
    if lambd_real is not None:
        (lambd_1_real, lambd_exp_real) = lambd_real
    _stem_est_real(ax, bact_all, 10**np.asarray(lambd_1)/24.,
                   None if lambd_real is None else 10**np.asarray(lambd_1_real)/24.)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Lag-Phasenübergangsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\lambda$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yscale('log')
    plt.savefig(path+'lambdas_1.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    _stem_est_real(ax, bact_all, lambd_exp, None if lambd_real is None else lambd_exp_real)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Lag-Phasenübergangsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\lambda$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yscale('log')
    plt.savefig(path+'lambdas_exp.png', bbox_inches='tight')
    plt.close(fig)

    temp = np.linspace(1.95, 14.05, 100)
    fig, ax = plt.subplots()
    for a0, a1, b in zip(lambd_1, lambd_exp, bact_all):
        #ax.plot(temp, lin_func(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2)
        ax.plot(temp, exp_func2(temp, a0, a1)/24., color=clrs1[b], linewidth=2)
    if lambd_real is not None:
        for a0, a1, b in zip(lambd_1_real, lambd_exp_real, bact_all):
            ax.plot(temp, exp_func2(temp, a0, a1)/24., color=clrs1[b], linewidth=2, linestyle='dashed')
    ax.set_title('Lag-Phasenübergangsrate', fontsize=14)
    ax.set_ylabel(r'$\lambda$ [$h^{-1}$]', fontsize=14)
    ax.set_xlabel(r'Temperatur [°C]', fontsize=14)
    ax.set_xlim(1.95, 14.05)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    _curve_legends(ax, bact_all, clrs1, lambd_real is not None)
    plt.savefig(path+'lambda_temp.png', bbox_inches='tight')
    plt.close(fig)


def plot_kij(kij_matrix, bact_all, path='', label=EST_LABEL, vmin=0.02*(5e-8), vmax=3.*(5e-8), cmap=None):
    """Visualize the pairwise bacterial interaction matrix; label: 'estimated', 'true', ... (title)."""
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    im = ax.imshow(kij_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, orientation='vertical')
    ax.set_title(r'Inhibitionskoeffizienten $k_{ij}$' + (f'\n({label})' if label else ''), fontsize=14, pad=14)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.set_yticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_yticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=12)
    print(path+'kij.png')
    plt.savefig(path+'kij.png', bbox_inches='tight')
    plt.close(fig)


def plot_x0(x0_vals, bact_all, exps, path='', x0_real=None):
    """Plot estimated (dots) and, if given, true (crosses) initial values by experiment and bacterium."""
    from matplotlib.lines import Line2D
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    ax.set_title('Ausgangskeimzahl ', fontsize=14)
    for i, exp in enumerate(exps):
        x0 = 10**np.array(x0_vals[n_cl*i:n_cl*(i+1)])
        ax.scatter(bact_all, x0, label=exp, color=plt_templ.exp_clrs[exp])
        if x0_real is not None:
            ax.scatter(bact_all, 10**np.array(x0_real[n_cl*i:n_cl*(i+1)]), marker='x', s=80, linewidths=2,
                       color=plt_templ.exp_clrs[exp])
    leg1 = ax.legend(fontsize=14, framealpha=0.1, bbox_to_anchor=(1, 1), loc='upper left')
    if x0_real is not None:
        ax.add_artist(leg1)
        ax.legend(handles=[Line2D([], [], marker='o', color='k', linestyle='none', label=EST_LABEL),
                           Line2D([], [], marker='x', color='k', markersize=9, markeredgewidth=2,
                                  linestyle='none', label=REAL_LABEL)],
                  fontsize=12, framealpha=0.1, bbox_to_anchor=(1, 0), loc='lower left')
    ax.set_yscale('log')
    ax.set_ylabel(r'$x_0$', fontsize=14)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(path+'init_vals.png', bbox_inches='tight')
    plt.close(fig)


def plot_parameters(param_ode, bact_all, exps, clrs1, param_real=[], path='', x0_real=None):
    """Plot the fitted ODE parameters and, when supplied, the true values
    (param_real: ODE part, x0_real: initial values, same layout as in param_ode)."""
    n_cl = len(bact_all)
    bact_all = [b if b!='Others' else 'Rest' for b in bact_all]
    #bact_all[-1] = 'Rest'
    param = param_ode[n_cl*len(exps):]

    #lambd_exp = param[:n_cl]#10**param[:n_cl]
    #alph0 = param[n_cl:2*n_cl]
    #alph1 = param[2*n_cl:3*n_cl]
    #nmax = param[3*n_cl]
    #kij_matrix = param[3*n_cl+1:].reshape(n_cl, -1)*(5e-8)

    lambd_1 = param[:n_cl]#10**param[:n_cl]
    lambd_exp = param[n_cl:2*n_cl]
    alph0 = param[2*n_cl:3*n_cl]
    alph1 = param[3*n_cl:4*n_cl]
    nmax = param[4*n_cl]
    nmax_exp = param[4*n_cl+1]
    kij_matrix = param[4*n_cl+2:].reshape(n_cl, -1)*(5e-8)
    x0_vals =  param_ode[:n_cl*len(exps)]

    print('nmax_1: ', nmax, 10**nmax, '\nnmax_exp: ', nmax_exp)
    plot_kij(kij_matrix, bact_all, path=path, label=EST_LABEL)
    plot_x0(x0_vals, bact_all, exps, path=path, x0_real=x0_real)
    if len(param_real)!=0:
        lambd_1_real = param_real[:n_cl]
        lambd_exp_real = param_real[n_cl:2*n_cl]
        alph0_real = param_real[2*n_cl:3*n_cl]
        alph1_real = param_real[3*n_cl:4*n_cl]

        plot_lambda((lambd_1, lambd_exp), bact_all, clrs1, lambd_real=(lambd_1_real, lambd_exp_real), path=path)
        plot_alpha((alph0, alph1), bact_all, clrs1, alph_real=(alph0_real, alph1_real), path=path)
        kij_real = np.asarray(param_real[4*n_cl+2:]).reshape(n_cl, -1)*(5e-8)
        plot_kij(kij_real, bact_all, path=path+'real_', label=REAL_LABEL)
        diff = kij_real - kij_matrix
        dmax = max(np.max(np.abs(diff)), 1e-30)
        plot_kij(diff, bact_all, path=path+'diff_', label=f'{REAL_LABEL} $-$ {EST_LABEL}',
                 vmin=-dmax, vmax=dmax, cmap='RdBu_r')
    else:
        plot_lambda((lambd_1, lambd_exp), bact_all, clrs1, path=path)
        plot_alpha((alph0, alph1), bact_all, clrs1, path=path)
    

def plot_cost_function(df_optim, path='', add_name=''):
    """Plot the optimization cost over recorded iterations."""
    fig, ax = plt.subplots()
    ax.plot(df_optim['iteration'], df_optim['cost'])
    ax.set_xlabel('optimization step', fontsize=12)
    ax.set_ylabel('cost function', fontsize=12)
    if np.all(np.asarray(df_optim['cost'], dtype=float) > 0):  # -2 log L objectives can be negative
        ax.set_yscale('log')
    plt.savefig(path+f'cost_plot{add_name}.png')#, bbox_inches='tight')
    plt.close(fig)
import matplotlib.pyplot as plt
import numpy as np
from . import plotting_templates as plt_templ


def lin_func(x, a0, a1):
    return a0 + a1*x

def exp_func(x, a0, a1):
    return a0*x**a1

def plot_alpha(alph, bact_all, clrs1, alph_real=None, path=''):
    temp = np.linspace(1.95, 14.05, 100)
    (alph_1, alph_exp) = alph
    fig, ax = plt.subplots()
    for a0, a1, b in zip(alph_1, alph_exp, bact_all):
        #ax.plot(temp, lin_func(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2)
        ax.plot(temp, exp_func(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2)
    if alph_real!=None:
        (alph_1_real, alph_exp_real) = alph_real
        for a0, a1, b in zip(alph_1_real, alph_exp_real, bact_all): 
            ax.plot(temp, exp_func(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2, linestyle='dashed')
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.set_xlabel(r'Temperatur [°C]', fontsize=14)
    ax.set_xlim(1.95, 14.05)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=14, framealpha=0.1, bbox_to_anchor=(1, 1))
    plt.savefig(path+'alphas.png', bbox_inches='tight')
    plt.close(fig)

    # Now plot separately alpha parameter values:
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    ax.stem(bact_all, alph_1)
    if alph_real!=None:
        ax.stem(bact_all, alph_1_real, markerfmt='x')
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(path+'alphas_1.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.stem(bact_all, alph_exp)
    if alph_real!=None:
        ax.stem(bact_all, alph_exp_real, markerfmt='x')
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Wachstumsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\alpha$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(path+'alphas_exp.png', bbox_inches='tight')
    plt.close(fig)


def exp_func2(x, a0, a1):
    return 10**a0 * x**a1

def plot_lambda(lambd, bact_all, clrs1, lambd_real=None, path=''):
    n_cl = len(bact_all)
    (lambd_1, lambd_exp) = lambd
    fig, ax = plt.subplots()
    ax.stem(bact_all, 10**lambd_1/24.)
    if lambd_real!=None:
        (lambd_1_real, lambd_exp_real) = lambd_real
        ax.stem(bact_all, 10**lambd_1_real/24., markerfmt='x')
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_title('Lag-Phasenübergangsrate', fontsize=14)
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'$\lambda$ [$h^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yscale('log')
    plt.savefig(path+'lambdas_1.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.stem(bact_all, lambd_exp)
    if lambd_real!=None:
        ax.stem(bact_all, lambd_exp_real, markerfmt='x')
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
        ax.plot(temp, exp_func2(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=2)
    if lambd_real!=None:
        (lambd_1_real, lambd_exp_real) = lambd_real
        for a0, a1, b in zip(lambd_1_real, lambd_exp_real, bact_all): 
            ax.plot(temp, exp_func2(temp, a0, a1)/24., label=b, color=clrs1[b], linewidth=3, linestyle='dashed')
    ax.set_title('Lag-Phasenübergangsrate', fontsize=14)
    ax.set_ylabel(r'$\lambda$ [$h^{-1}$]', fontsize=14)
    ax.set_xlabel(r'Temperatur [°C]', fontsize=14)
    ax.set_xlim(1.95, 14.05)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=14, framealpha=0.1, bbox_to_anchor=(1, 1))
    plt.savefig(path+'lambda_temp.png', bbox_inches='tight')
    plt.close(fig)


def plot_kij(kij_matrix, bact_all, path=''):
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    im = ax.imshow(kij_matrix, vmin=0.02*(5e-8), vmax=3.*(5e-8))
    fig.colorbar(im, orientation='vertical')
    ax.set_title(r'Inhibitionskoeffizienten $k_{ij}$', fontsize=14)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.set_yticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_yticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=12)
    print(path+'kij.png')
    plt.savefig(path+'kij.png', bbox_inches='tight')
    plt.close(fig)


def plot_x0(x0_vals, bact_all, exps, path=''):
    n_cl = len(bact_all)
    fig, ax = plt.subplots()
    ax.set_title('Ausgangskeimzahl ', fontsize=14)
    for i, exp in enumerate(exps):
        x0 = 10**np.array(x0_vals[n_cl*i:n_cl*(i+1)])
        ax.scatter(bact_all, x0, label=exp, color=plt_templ.exp_clrs[exp])
    ax.legend(fontsize=14, framealpha=0.1, bbox_to_anchor=(1, 1))
    ax.set_yscale('log')
    ax.set_ylabel(r'$x_0$', fontsize=14)
    ax.set_xticks(np.linspace(0, n_cl-1, n_cl))
    ax.set_xticklabels(bact_all, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(path+'init_vals.png', bbox_inches='tight')
    plt.close(fig) 


def plot_parameters(param_ode, bact_all, exps, clrs1, param_real=[], path=''):
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
    plot_kij(kij_matrix, bact_all, path=path)
    plot_x0(x0_vals, bact_all, exps, path=path)
    if len(param_real)!=0:
        lambd_1_real = param_real[:n_cl]
        lambd_exp_real = param_real[n_cl:2*n_cl]
        alph0_real = param_real[2*n_cl:3*n_cl]
        alph1_real = param_real[3*n_cl:4*n_cl]

        plot_lambda((lambd_1, lambd_exp), bact_all, clrs1, lambd_real=(lambd_1_real, lambd_exp_real), path=path)
        plot_alpha((alph0, alph1), bact_all, clrs1, alph_real=(alph0_real, alph1_real), path=path)
        plot_kij(param_real[4*n_cl+2:].reshape(n_cl, -1)*(5e-8), bact_all, path=path+'real_')
        plot_kij(param_real[4*n_cl+2:].reshape(n_cl, -1)*(5e-8)-kij_matrix, bact_all, path=path+'diff_')
    else:
        plot_lambda((lambd_1, lambd_exp), bact_all, clrs1, path=path)
        plot_alpha((alph0, alph1), bact_all, clrs1, path=path)
    

def plot_cost_function(df_optim, path='', add_name=''):
    fig, ax = plt.subplots()
    ax.plot(df_optim['iteration'], df_optim['cost'])
    ax.set_xlabel('optimization step', fontsize=12)
    ax.set_ylabel('cost function', fontsize=12)
    ax.set_yscale('log')
    plt.savefig(path+f'cost_plot{add_name}.png')#, bbox_inches='tight')
    plt.close(fig)
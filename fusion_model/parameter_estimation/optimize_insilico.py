import numpy as np
from .. import model as mdl

def param_bnds_insilico_tempexponent(n_cl, inhib=True, lambda_exp_max=5., lambda_exp_min=0., fix_nexp=False,
                                     lambda_1_bounds=(-8., -1.)):
    """Build ODE parameter bounds for the temperature-exponent in-silico model (fix_nexp: N_exp fixed at 0)."""
    if inhib == True:  # noqa: E712
        inhib_bnds = [(0.0, 20.) if i!=j else (0., 0.)
                  for i in range (n_cl)
                  for j in range (n_cl)]
    else:
        inhib_bnds = [(0., 0.) for _ in range (n_cl*n_cl)]
    param_ode_bnds = tuple([tuple(lambda_1_bounds) for _ in range (n_cl)]  + # lambd_1
                            [(lambda_exp_min, lambda_exp_max) for _ in range (n_cl)] + # lambd_exp
                            [(.001,  2.)  for _ in range (n_cl)] +  # mu_0
                            [(.0,  2.)  for _ in range (n_cl)] +  # mu_exp
                            [(7.5, 14.)] + [(0., 0. if fix_nexp else 2.)] + # N_1 and N_exp
                            inhib_bnds)
    
    return param_ode_bnds


def param_bnds_insilico_templinear(n_cl, inhib=True):
    """Build ODE parameter bounds for the linear-temperature in-silico model."""
    if inhib == True:  # noqa: E712
        inhib_bnds = [(0.02, 3.) if i!=j else (0., 0.)
                  for i in range (n_cl)
                  for j in range (n_cl)]
    else:
        inhib_bnds = [(0., 0.) for _ in range (n_cl*n_cl)]
    param_ode_bnds = tuple([(-7., -2.)  for _ in range (n_cl)] + # lambd_1
                            [(.01,  1.) for _ in range (n_cl)] + # mu_0
                            [(.1 ,  .6) for _ in range (n_cl)] + # mu_1
                            [(7.5, 14.)] +      
                            inhib_bnds)
    return param_ode_bnds


def define_calibr_setup_insilico(calibr_presetup, inhib=True, s_x_predefined=None, s_x=None, lambda_exp_max=5.,
                                 lambda_exp_min=0., x0_bounds=(0., 5.), fix_nexp=False, lambda_1_bounds=(-8., -1.)):
    """Add model-specific parameter bounds and optional media weights to an in-silico setup.
    lambda_1_bounds: bounds of lambda_1 (log10 lag rate at 1 degC); lambda_exp_min / lambda_exp_max: bounds of lambda_exp, fix_nexp: N_exp fixed at 0 (fusion_model2);
    x0_bounds: bounds of the log10 initial values."""
    x0_bnds_all = tuple([tuple(x0_bounds) for _ in range (calibr_presetup['n_cl']) for _ in range (len(calibr_presetup['exps']))])
    calibr_setup = calibr_presetup
    if calibr_presetup['model'] in (mdl.fusion_model2, mdl.fusion_model2_f1, mdl.fusion_model2_f2):
        calibr_setup['param_bnds'] =  x0_bnds_all + param_bnds_insilico_tempexponent(calibr_presetup['n_cl'], inhib=inhib,
                                                                                    lambda_exp_max=lambda_exp_max,
                                                                                    lambda_exp_min=lambda_exp_min,
                                                                                    fix_nexp=fix_nexp,
                                                                                    lambda_1_bounds=lambda_1_bounds)
    elif calibr_presetup['model'] == mdl.fusion_model_linear:
        calibr_setup['param_bnds'] =  x0_bnds_all + param_bnds_insilico_templinear(calibr_presetup['n_cl'], inhib=inhib)
    else:
        print('Unknown model!')
        exit()

    if s_x is not None:
        calibr_setup['s_x'] = s_x
    else:
        if s_x_predefined is not None:
            S_bnds = []
            for i, s in enumerate(s_x_predefined.flatten()):
                if np.isnan(s):
                    S_bnds.append((0.01, 1.)) 
                else:
                    S_bnds.append((float(s), float(s)))
        else: 
            S_bnds = [(0.01, 1.) for _ in range (calibr_presetup['n_cl']*len(calibr_presetup['exps']))]
        calibr_setup['param_bnds'] = calibr_setup['param_bnds'] + tuple(S_bnds)
    return calibr_setup
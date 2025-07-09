import numpy as np
from .. import model as mdl

def param_bnds_ZL2030_tempexponent(n_cl, inhib=True):
    if inhib == True:  # noqa: E712
        inhib_bnds = [(0.02, 3.) if i!=j else (0., 0.)
                  for i in range (n_cl)
                  for j in range (n_cl)]
    else:
        inhib_bnds = [(0., 0.) for _ in range (n_cl*n_cl)]
    param_ode_bnds = tuple([(-7., -2.)  for _ in range (n_cl)]    + # lambd_1
                            [(0., 3.)    for _ in range (n_cl)]   + # lambd_exp
                            [(.01,  1.)  for _ in range (n_cl-1)] + [(0., 0)] + # alph_0
                            [(.01,  2.)  for _ in range (n_cl-1)] + [(0., 0)] + # alph_exp
                            [(7.5, 14.)] + [(0., 2.)]             + # N_1 and N_exp
                            inhib_bnds)
    
    return param_ode_bnds


def param_bnds_ZL2030_templinear(n_cl, inhib=True):
    if inhib == True:  # noqa: E712
        inhib_bnds = [(0.02, 3.) if i!=j else (0., 0.)
                  for i in range (n_cl)
                  for j in range (n_cl)]
    else:
        inhib_bnds = [(0., 0.) for _ in range (n_cl*n_cl)]
    param_ode_bnds = tuple([(-7., -2.)  for _ in range (n_cl)]   + # lambd_1
                            [(.01,  1.) for _ in range (n_cl-1)] + [(0., 0)] + # alph_0 # Others  do not grow
                            [(.2 ,  .6) for _ in range (n_cl-1)] + [(0., 0)] + # alph_1 # Others  do not grow
                            [(7.5, 14.)] +      
                            inhib_bnds), 
    return param_ode_bnds


def define_calibr_setup_ZL2030(calibr_presetup, inhib=True, s_x_predefined=None, s_x=None):
    x0_bnds_all = tuple([(1., 4.5) for _ in range (calibr_presetup['n_cl']) for _ in range (len(calibr_presetup['exps']))])
    calibr_setup = calibr_presetup
    if calibr_presetup['model'] == mdl.fusion_model2:
        calibr_setup['param_bnds'] =  x0_bnds_all + param_bnds_ZL2030_tempexponent(calibr_presetup['n_cl'], inhib=inhib)
    elif calibr_presetup['model'] == mdl.fusion_model_linear:
        calibr_setup['param_bnds'] =  x0_bnds_all + param_bnds_ZL2030_templinear(calibr_presetup['n_cl'], inhib=inhib)
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
                    S_bnds.append((0.05, 1.)) 
                else:
                    S_bnds.append((float(s), float(s)))
        else: 
            S_bnds = [(0.05, 1.) for _ in range (calibr_presetup['n_cl']*len(calibr_presetup['exps']))]
        calibr_setup['param_bnds'] = calibr_setup['param_bnds'] + tuple(S_bnds)
    return calibr_setup
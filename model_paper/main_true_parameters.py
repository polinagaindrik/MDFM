"""
True parameters of the in-silico model from the ZL2030 estimates
================================================================

Run once (separately from the calibration) to (re)create
    model_paper/setup/True_parameters_paper.json
which all in-silico data generation reads (fm.data.read_true_parameters, model_exp):

  param_ode = [x0 (N_EXPS x N_CL) | lambda_1 | lambda_exp | alpha_0 | alpha_exp | N_1, N_exp | k_ij (N_CL x N_CL)]
              x0                : first N_EXPS experiments of setup/Initial_values_x0_paper.json
              ODE parameters    : ZL2030 calibration (exponential model), rounded to one decimal, k_ii = 0
  s_x       = [first selective medium, first general medium] of setup/Media_matrix_S_paper.json
  T_x       = T_x of setup/Media_matrix_S_paper.json

The setup files themselves (initial values, media matrix) come from main_insilicomodel_generation.py.
An existing file with different content is kept as True_parameters_paper_previous.json.

Run from the repository root:
    python model_paper/main_true_parameters.py
"""
import os
import sys
import shutil
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np


if __name__ == "__main__":
    # ================================================================
    ZL2030_DIR = 'out/zl2030/exp_model/calibration/'   # folder of Result_calibration_zl2030_copy.json
    SETUP_DIR = fm.data.SETUP_DIR                       # model_paper/setup/
    OUT_FILE = fm.data.TRUE_PARAMETERS_FILE             # True_parameters_paper.json
    N_CL = 12      # species
    N_EXPS = 9     # experiments (initial values)
    # ================================================================

    res = fm.data.true_parameters_from_zl2030(ZL2030_DIR, n_cl=N_CL, n_exps=N_EXPS, setup_dir=SETUP_DIR)

    out = os.path.join(SETUP_DIR, OUT_FILE)
    if os.path.exists(out):
        old = fm.data.read_from_json(OUT_FILE, dir=SETUP_DIR)
        same = all(np.allclose(np.asarray(old[k], dtype=float), np.asarray(res[k], dtype=float))
                   for k in ('param_ode', 's_x', 'T_x') if len(old[k]) == len(res[k])) and \
            all(len(old[k]) == len(res[k]) for k in ('param_ode', 's_x', 'T_x'))
        if same:
            print(f'{out} is up to date (identical content).')
        else:
            bak = out.replace('.json', '_previous.json')
            shutil.copy(out, bak)
            print(f'{out} differs from the new values: previous version kept as {bak}')
    fm.data.json_dump(res, OUT_FILE, dir=SETUP_DIR)

    tp = fm.data.read_true_parameters(N_CL, dir=SETUP_DIR)
    n = N_CL
    o = tp['param_ode']
    print(f'Saved {out}: {N_EXPS} experiments x {N_CL} species')
    print('  lambda_1  :', o[:n])
    print('  lambda_exp:', o[n:2*n])
    print('  alpha_0   :', o[2*n:3*n])
    print('  alpha_exp :', o[3*n:4*n])
    print('  N_1, N_exp:', o[4*n:4*n+2])
    print('  s_x       :', tp['s_x'].tolist())

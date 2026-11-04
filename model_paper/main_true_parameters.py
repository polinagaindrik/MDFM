"""
True parameters of the in-silico model for an initial-value distribution
=========================================================================

Run once (separately from the calibration) after main_insilicomodel_generation.py to create
    model_paper/setup/True_parameters_paper_<X0_DISTRIBUTION>.json   (e.g. lognormal)
which the in-silico data generation reads (fm.data.read_true_parameters):

  param_ode = [x0 (N_EXPS x 12) | lambda_1 | lambda_exp | alpha_0 | alpha_exp | N_1, N_exp | k_ij (12 x 12)]
              x0                : first N_EXPS experiments of setup/Initial_values_x0_paper_<X0_DISTRIBUTION>.json
              ODE parameters    : copied from the frozen setup/True_parameters_paper_uniform.json
  s_x       = [first selective medium, first general medium] of setup/Media_matrix_S_paper.json
  T_x       = T_x of setup/Media_matrix_S_paper.json

True_parameters_paper_uniform.json is the fixed reference (not rewritten by this script), so the ODE
parameters are the same for all distributions; s_x and T_x come from the media matrix. An existing output file with
different content is kept as True_parameters_paper_<X0_DISTRIBUTION>_previous.json.

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
    SOURCE = 'uniform'  # keep 'uniform': ODE parameters are copied from the frozen True_parameters_paper_uniform.json
    SETUP_DIR = fm.data.SETUP_DIR                       # model_paper/setup/
    X0_DISTRIBUTION = 'lognormal'                       # initial values of the new file (not SOURCE)
    OUT_FILE = fm.data.true_parameters_file(X0_DISTRIBUTION)  # True_parameters_paper_<X0_DISTRIBUTION>.json
    N_EXPS = None  # experiments (initial values); None: as many as in the source file (9)
    # ================================================================
    if X0_DISTRIBUTION == SOURCE:
        raise ValueError(f'{fm.data.true_parameters_file(SOURCE)} is the frozen reference and is not rewritten; '
                         f'choose another X0_DISTRIBUTION')

    res = fm.data.true_parameters_for_distribution(X0_DISTRIBUTION, source=SOURCE, n_exps=N_EXPS, setup_dir=SETUP_DIR)
    N_CL = len(res['T_x'])

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

    tp = fm.data.read_true_parameters(N_CL, file=OUT_FILE, dir=SETUP_DIR)
    n = N_CL
    o = tp['param_ode']
    print(f'Saved {out}: {len(tp["x10"])} experiments x {N_CL} species (ODE parameters from '
          f'{fm.data.true_parameters_file(SOURCE)}, s_x and T_x from {fm.data.MEDIA_MATRIX_FILE})')
    print('  lambda_1  :', o[:n])
    print('  lambda_exp:', o[n:2*n])
    print('  alpha_0   :', o[2*n:3*n])
    print('  alpha_exp :', o[3*n:4*n])
    print('  N_1, N_exp:', o[4*n:4*n+2])
    print('  s_x       :', tp['s_x'].tolist())

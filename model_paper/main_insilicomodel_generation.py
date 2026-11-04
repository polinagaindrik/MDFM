import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np

if __name__ == "__main__":
   np.random.seed(4698517)
   path = fm.data.SETUP_DIR  # model_paper/setup/; afterwards run main_true_parameters.py (same X0_DISTRIBUTION)
   n_exp_max = 15
   n_cl_max = 12
   add_name = '_paper'
   # distribution of the initial values L0 = log10(x0) (see fm.data.sample_log10_x0), saved to
   # setup/Initial_values_x0_paper_<X0_DISTRIBUTION>.json (choose the file in the calibration scripts by X0_DISTRIBUTION):
   #   'uniform'  : L0 ~ U(1, 4), x0 log-uniform (the paper setup)
   #   'lognormal': L0 ~ N(X0_MEAN, X0_SIGMA) truncated to X0_RANGE, x0 lognormal; X0_MEAN, X0_SIGMA: maximum-likelihood
   #                fit to the 180 log10 initial values of the ZL2030 calibration (out/zl2030/exp_model/calibration/),
   #                X0_RANGE = [1, 4.5]: bounds of x0 in the ZL2030 calibration
   # The media matrix setup/Media_matrix_S_paper.json (shared by all distributions) is only read, never overwritten.
   # SELECTIVE_MEDIA = 'generate': N_SELECTIVE new selective media (fm.data.sample_selective_media: each medium
   # targets its own 3 species, no species is the target of two media, targets of one medium at least 4
   # apart in species index, recovery 0.6-0.95, all other species leakage 0.05-0.1) with seed SEED_S, saved with the
   # general media of the read file to setup/Media_matrix_S_paper_generated.json (to use it, replace
   # Media_matrix_S_paper.json by it).
   # GENERAL_MEDIA = 'generate': gen0 = general medium estimated from the ZL2030 data (plate count agar, key
   # 's_general0') and gen1..gen<N_GENERAL> (rows of 's_general') = random permutations of gen0 over the species
   # (fm.data.sample_general_media), saved to the same file. Pairing gen k + sel k: the low values (< 0.15) of gen k
   # are placed on the target species of sel k, so every species is visible in each gen k + sel k setup
   # (needs SELECTIVE_MEDIA = 'generate', or the targets are taken from the read selective media).
   # The selective media are checked against these rules in both cases (fm.data.check_selective_media).
   X0_DISTRIBUTION = 'lognormal'
   X0_MEAN = 2.9
   X0_SIGMA = 0.69
   X0_RANGE = (1., 4.5)  # lognormal only; uniform: (1, 4)
   SELECTIVE_MEDIA = 'generate'   # 'read' or 'generate'
   GENERAL_MEDIA = 'generate'     # 'read' or 'generate'
   N_GENERAL = 3
   SEED_S = 12345
   N_SELECTIVE = 3
   x10 = fm.data.generate_x0_for_simulation(n_exp_max, n_cl_max, path=path, add_name=f'{add_name}_{X0_DISTRIBUTION}',
                                            distribution=X0_DISTRIBUTION, mean=X0_MEAN, sigma=X0_SIGMA,
                                            **(dict(zip(('low', 'high'), X0_RANGE)) if X0_DISTRIBUTION == 'lognormal' else {}))
   S_matrix_setup = fm.data.read_from_json(f'Media_matrix_S{add_name}.json', dir=path)
   np.random.seed(SEED_S)
   if SELECTIVE_MEDIA == 'generate':
      s_selective, groups = fm.data.sample_selective_media(n_media=N_SELECTIVE, n_cl=n_cl_max)
      S_matrix_setup = dict(S_matrix_setup, s_selective=s_selective)
      print(f'Selective media generated (target species {groups})')
   if GENERAL_MEDIA == 'generate':
      pair_targets = [[int(k) for k in np.flatnonzero(np.asarray(r) >= 0.6)] for r in S_matrix_setup['s_selective']]
      s_general0, s_general = fm.data.sample_general_media(n_media=N_GENERAL, n_cl=n_cl_max, pair_targets=pair_targets)
      S_matrix_setup = dict(S_matrix_setup, s_general0=s_general0, s_general=s_general)
      print('General media generated: gen0 = ZL2030 plate count medium, gen1-gen3 = its permutations')
   if 'generate' in (SELECTIVE_MEDIA, GENERAL_MEDIA):
      fm.data.json_dump(S_matrix_setup, path + f'Media_matrix_S{add_name}_generated.json')
      print(f'Saved to {path}Media_matrix_S{add_name}_generated.json (replace Media_matrix_S{add_name}.json by it to use it)')
   problems = fm.data.check_selective_media(S_matrix_setup['s_selective'])
   print('Selective media: ' + ('all rules fulfilled' if not problems else 'problems:\n  ' + '\n  '.join(problems)))

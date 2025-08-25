import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import numpy as np

def generate_S_matrix(n_cl_max, add_name=''):
   s_general = np.round(np.random.uniform(0.0, 1.0, size=(n_cl_max, n_cl_max)), 1)
   s_selective = np.round(np.random.uniform(0.0, 05.0, size=(n_cl_max, n_cl_max)) + np.random.uniform(0.6, 0.95, n_cl_max)*np.eye(n_cl_max), 1)
   s_x, T_x, param_ode = fm.data.get_res_from_zl2030dict(n_cl_max, model='exponential', dir='out/zl2030/exp_model/calibration/')

   S_matrix_setup = {'s_general': s_general,
                     's_selective': s_selective,
                     's_general_zl2030': s_x[1],
                     's_selective_zl2030': s_x[0],
                     'T_x': T_x} 
   fm.data.json_dump(S_matrix_setup, path+f'Media_matrix_S{add_name}.json')
   return S_matrix_setup

if __name__ == "__main__":
   np.random.seed(4698517)
   path = 'model_paper/out/'
   n_exp_max = 15
   n_cl_max = 12
   add_name = '_paper'
   x10 = fm.data.generate_x0_for_simulation(n_exp_max, n_cl_max, path=path, add_name=add_name)
   S_matrix_setup = generate_S_matrix(n_cl_max, add_name=add_name) 
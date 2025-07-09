import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import chi2
from .optimization import optimization_func


# Calculate joint probability array from 1D undependent probabilities along each parameter
def get_joint_distrib_val_from_projections(index, param_arr, prob_projections):
    param = np.array([param_arr[j][index[j]] for j in range (len(index))])
    prob = np.prod([prob_projections[j][index[j]] for j in range (len(index))])
    return [index, param, prob]


# Calculate the 1D probabilities along 1 parameter with index param_num given a joint probability of all parameters
# Integrate out all other parameters except param_num (used to test obtained joint pribability)
def get_prob_proj_from_joint(param_arr, prob_arr, param_num):
    change_axis = 0
    for i, p in enumerate(param_arr):
        if i != param_num:
            new_prob = trapezoid(prob_arr, p, axis=change_axis)
            prob_arr = new_prob
        else:
            change_axis += 1
            new_prob = prob_arr
    return new_prob


# Define parameter values to calculate probabilities give bounds and number of points
def get_param_array(param_bnds, num_points):
    return np.array([np.linspace(bnd[0], bnd[1], num_points) for bnd in param_bnds])


def calculate_likelihood(ll_func, calibr_setup, dfs, param_opt, num_points=10):
    param_arr = get_param_array(calibr_setup['param_bnds'], num_points)
    ll_arr = np.zeros(np.shape(param_arr))
    n_param = len(param_arr)
    for i in range (n_param):
        print(i)
        for k, p_val in enumerate(param_arr[i]):
            ll_arr[i, k] = ll_func(np.insert(param_opt, i, p_val), dfs, calibr_setup['model'], calibr_setup['P_matrix'], calibr_setup['s_x'])
    return param_arr, ll_arr


def calculate_profile_likelihood(func_ll, calibr_setup, dfs, num_points=10):
    param_arr = get_param_array(calibr_setup['param_bnds'], num_points)
    profll_arr = np.zeros(np.shape(param_arr))
    n_param = len(param_arr)
    for i in range (n_param):
        print(i)
        bnds_pfixed = [b for j, b in enumerate(calibr_setup['param_bnds']) if j != i]
        for k, p_val in enumerate(param_arr[i]):
            profll_arr[i, k] = optimization_func(func_for_profll, bnds_pfixed,
                                                 args=(func_ll, p_val, i, dfs, calibr_setup['model'], calibr_setup['P_matrix'], calibr_setup['s_x']),
                                                 workers=calibr_setup['workers']).fun
    return param_arr, profll_arr


def func_for_profll(func_ll, p, p_fix, ind_param, dfs, model, Pmatrices, s_x):
    return func_ll(np.insert(p, ind_param, p_fix), dfs, model, Pmatrices, s_x)


def confidence_intervals(param_arr, profll_arr, confidence_level=0.95):
    threshold = chi2.ppf(confidence_level, 1)
    profll_opt = np.min(profll_arr)
    CI = []
    for p, prof in zip(param_arr, profll_arr):
        allind = np.where(prof-profll_opt <= threshold)[0]
        CI.append((p[np.max(allind[0]-1, 0)], 
                   p[np.min([allind[-1]+1, len(p)])]))
    return CI
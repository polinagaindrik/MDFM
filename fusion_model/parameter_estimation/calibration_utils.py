"""
Helpers around a calibration that work the same for experimental (ZL2030) and in-silico data: names of the
parameter vector, information read from the data frames, fixed parameters and reloading a saved calibration.

Parameter vector layout (cost_withS, model fusion_model2):
    [ x0 (n_cl per experiment, log10) | lambda_1 (n_cl) | lambda_exp (n_cl) | mu_0 (n_cl) | mu_exp (n_cl) |
      N_1 | N_exp | k_ij (n_cl x n_cl) | S (n_media x n_cl) ]
"""
import glob
import os

import numpy as np
import pandas as pd

from .. import model as mdl
from ..data.output import read_from_json
from ..tools.dataframe_functions import extract_observables_from_df
from . import profile_likelihood as pl
from .cost_function import cost_sum_and_geometric_mean
from .optimization import apply_fixed_values

__all__ = ['load_calibration', 's_x_predefined_from_maldi', 'exp_temps_from_columns', 'make_param_names',
           'resolve_fixed_values', 'fixed_values_from_result', 'print_parameter_table', 'copy_result_to_all_profiles',
           'build_calibr_setup']


def load_calibration(path, add_name):
    """Data frames, estimated parameters and the true parameters of in-silico data (None for experimental data)."""
    dfs = []
    for k in ["platecount", "maldi", "ngs"]:
        f_pre = os.path.join(path, f"dataframe_{k}{add_name}_preprocessed.pkl")
        f_raw = os.path.join(path, f"dataframe_{k}{add_name}.pkl")
        dfs.append(pd.read_pickle(f_pre if os.path.exists(f_pre) else f_raw))
    result = read_from_json(f"Result_calibration{add_name}.json", dir=path)
    true_params = None
    if os.path.exists(os.path.join(path, "Result_temp_together_real.json")):
        tr = read_from_json("Result_temp_together_real.json", dir=path)
        true_params = np.concatenate([np.ravel(tr["param_ode"]), np.ravel(tr["s_x"])])
    return dfs, result, true_params


def s_x_predefined_from_maldi(df_maldi, media):
    """Same rule as dtf.make_df_maldi_ngs_compatible: S fixed to 0 for bacteria
    never seen on a medium, free (NaN) otherwise."""
    s = np.full((len(media), np.shape(df_maldi)[0]), np.nan)
    for i, med in enumerate(media):
        zero = (df_maldi.filter(like=med).T.sum() == 0).to_numpy()
        s[i, zero] = 0.
    return s


def exp_temps_from_columns(df):
    """{exp: temperature} parsed from column names like 'V01_M1_02C_00_MRS-platecount'."""
    temps = {}
    for c in df.columns:
        parts = c.split("_")
        temps.setdefault(parts[0], float(parts[2].rstrip("C")))
    return temps


def make_param_names(exps, bact, media):
    """LaTeX name of every entry of the parameter vector."""
    n_cl = len(bact)
    b = [s.split("_")[-1] for s in bact]  # 'Bacteria_03' -> '03'
    names = [rf"$x_{{0,{bi}}}^{{{e}}}$" for e in exps for bi in b]
    names += [rf"$\lambda_{{1,{bi}}}$" for bi in b]
    names += [rf"$\lambda_{{exp,{bi}}}$" for bi in b]
    names += [rf"$\mu_{{0,{bi}}}$" for bi in b]
    names += [rf"$\mu_{{exp,{bi}}}$" for bi in b]
    names += [r"$N_1$", r"$N_{exp}$"]
    names += [rf"$k_{{{b[i]},{b[j]}}}$" for i in range(n_cl) for j in range(n_cl)]
    names += [rf"$S_{{{m},{bi}}}$" for m in media for bi in b]
    return names


def resolve_fixed_values(fix, names):
    """{index: value} from a dict whose keys are indices or parameter names (as in make_param_names,
    with or without the LaTeX $ \\ { }), e.g. {17: 0.} or {'k_00,01': 0., 'mu_exp,00': 0.}."""
    return {pl._resolve_param_index(k, names): float(v) for k, v in (fix or {}).items()}


def fixed_values_from_result(result):
    """{index: value} stored by the refinement as 'fixed_params' in Result_calibration*.json."""
    return {int(i): float(d["value"]) for i, d in (result.get("fixed_params") or {}).items()}


def print_parameter_table(names, param, bnds, fixed=None):
    """Index, name, value and bounds of every parameter (to choose indices for FIX_PARAMS)."""
    fixed = fixed or {}
    print("Parameters (index, name, value, bounds):")
    for i, (n, v, b) in enumerate(zip(names, param, bnds)):
        tag = f"   fixed at {fixed[i]:g}" if i in fixed else ("   (fixed by bounds)" if b[1] <= b[0] else "")
        print(f"  {i:3d}  {n:<28} {v:12.5g}   ({b[0]:g}, {b[1]:g}){tag}")


def copy_result_to_all_profiles(path, add_name, profile_dir="profile_likelihood"):
    """
    After a new local minimization: copy param_ode of <path>/Result_calibration<add_name>.json into
    every existing profile run of this calibration (<path>/<profile_dir>/profile_likelihood_results
    <add_name>_*_summary.json / _run.json), see pl.copy_result_to_profile_json. Runs with a different
    number of parameters are skipped. cost_opt and the profile points are not changed.
    """
    result_json = os.path.join(path, f"Result_calibration{add_name}.json")
    folder = os.path.join(path, profile_dir)
    bases = set()
    for suffix in ("_summary.json", "_run.json"):
        for f in glob.glob(os.path.join(folder, f"profile_likelihood_results{add_name}_*{suffix}")):
            if not f.endswith(("_before_copy.json", "_before_update.json")):
                bases.add(f[:-len(suffix)])
    if not bases:
        print(f"No profile likelihood results in {folder}: nothing to update.")
        return []
    done = []
    for b in sorted(bases):
        try:
            pl.copy_result_to_profile_json(result_json, b + ".csv")
            done.append(b + ".csv")
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipped {b}: {e}")
    return done


def build_calibr_setup(path, add_name, n_cl, n_media, define_setup, objective="neg2logL", exps=None):
    """calibr_setup (incl. param_bnds and data_array) for a saved calibration.
    define_setup: function that adds the parameter bounds of this kind of data, e.g. fm.pest.define_calibr_setup_insilico
    objective: 'neg2logL'    GaussianNeg2LogLik: -2 log L, data types weighted by their number of points
               'geomean'     GeometricMeanNeg2LogLik: -2 log L version of the calibration cost
                             (same optimum as cost_sum_and_geometric_mean, equal weight per data type)
               'calibration' cost_sum_and_geometric_mean itself (scale='auto': linearized G_opt / N)
    exps: experiments of the calibration; None: all experiments of the saved plate count data frame. Experimental
          data frames can contain more experiments than were used in the calibration: then give exps, the data
          frames are reduced to them."""
    if objective not in ("neg2logL", "geomean", "calibration"):
        raise ValueError(f"unknown objective {objective!r}")
    dfs, result, true_params = load_calibration(path, add_name)
    if exps is not None:  # the cost function takes the experiments from the columns of the data frames
        dfs = [df[[c for c in df.columns if c.split("_")[0] in exps]] for df in dfs]
    df_platecount, df_maldi, df_ngs = dfs
    exps = sorted(list(set([s.split("_")[0] for s in df_platecount.columns])))
    media = sorted(list(set([s.split("_")[-1].split("-")[0] for s in df_maldi.columns])))
    bact = list(df_maldi.index)
    assert len(bact) == n_cl and len(media) == n_media, (bact, media)
    calibr_presetup = {
        "model": mdl.fusion_model2,
        "T_x": result["T_x"],
        "workers": 1,
        "output_path": path,
        "n_cl": n_cl,
        "n_media": n_media,
        "dfs": dfs,
        "aggregation_func": cost_sum_and_geometric_mean,
        "exps": exps,
        "exp_temps": exp_temps_from_columns(df_platecount),
        "media": media,
    }
    calibr_setup = define_setup(calibr_presetup, inhib=True, s_x_predefined=s_x_predefined_from_maldi(df_maldi, media),
                                s_x=None)
    calibr_setup["data_array"] = extract_observables_from_df(dfs)
    calibr_setup["s_max_penalty"] = result.get("s_max_penalty")  # same penalty as in the calibration
    if calibr_setup["s_max_penalty"]:
        print(f"Calibration used s_max_penalty = {calibr_setup['s_max_penalty']}: applied in the profiles too")
    calibr_setup["noise_model"] = result.get("noise_model", "absolute")  # same noise model as the calibration
    print(f"Noise model of the calibration: {calibr_setup['noise_model']}")
    calibr_setup["maldi_n"] = result.get("maldi_n")  # same MALDI likelihood as the calibration
    calibr_setup["ngs_dm"] = result.get("ngs_dm")  # same NGS likelihood as the calibration
    if objective == "neg2logL":
        calibr_setup["aggregation_func"] = pl.make_gaussian_neg2loglik(calibr_setup["data_array"],
                                                                       calibr_setup["noise_model"],
                                                                       maldi_n=calibr_setup["maldi_n"],
                                                                       ngs_dm=calibr_setup["ngs_dm"])
    elif objective == "geomean":
        calibr_setup["aggregation_func"] = pl.make_geomean_neg2loglik(calibr_setup["data_array"],
                                                                      calibr_setup["noise_model"])
    param_opt = np.array(result["param_ode"], dtype=float)
    param_names = make_param_names(exps, bact, media)
    assert len(param_opt) == len(calibr_setup["param_bnds"]) == len(param_names)
    fixed = fixed_values_from_result(result)
    if fixed:  # parameters held fixed in the refinement stay fixed (not profiled, not re-fitted)
        calibr_setup, param_opt = apply_fixed_values(calibr_setup, fixed, param=param_opt, names=param_names)
    return calibr_setup, param_opt, param_names, true_params

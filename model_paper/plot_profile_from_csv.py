"""
Plot profile likelihood results from a saved CSV (model_paper)
==============================================================

Counterpart of ``pool_paper_casestudy/plot_profile_from_csv.py`` (in the
separate pool-paper-casestudy repo, https://github.com/polinagaindrik/pool-paper-casestudy): reads a CSV
produced by run_profile_likelihood_all() (columns: param_index, param_value,
cost, <one column per parameter>) and regenerates the plots without
recomputing anything. The scale is read from <csv>_summary.json (scale='auto'),
or pass it explicitly as in pool_paper. While the profile run is still going
(final CSV not written yet), the finished walks in <csv>_partial.csv are plotted.

Run from the repository root:
    python model_paper/plot_profile_from_csv.py
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fusion_model.parameter_estimation.profile_likelihood import (  # noqa: E402
    plot_profile_likelihood_from_csv, plot_profile_likelihood_individual)
from profile_likelihood_model_paper import load_calibration  # noqa: E402
import paper_paths as pp  # noqa: E402


if __name__ == "__main__":
    # ================================================================
    # EDIT THESE, THEN JUST RUN THIS FILE (no command-line args needed)
    # ================================================================
    # STUDY: 'test_2sp' (DESIGN as in test_2sp.py) or 'model_complexity', 'noise_vs_nspecies', 'media_influence'
    # (MEDIA), folders see paper_paths.py
    STUDY = "test_2sp"
    DESIGN = "base"
    n_cl, n_media, relnoise = 2, 2, 0.1
    MEDIA = None  # media_influence only, e.g. "gen2+sel2" (sets n_media)
    if STUDY == "test_2sp":
        add_name = f"_{int(n_cl)}dim_{int(n_media)}media_test"
        path2 = pp.test_2sp_path(DESIGN, relnoise)
    else:
        if MEDIA is not None:
            n_media = pp.media_n(MEDIA)
        add_name = pp.add_name(n_cl, n_media, media=MEDIA)
        path2 = pp.study_path(STUDY, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=MEDIA)
    tag = f"{add_name}_all_neg2logL"
    CSV_PATH = path2 + f"profile_likelihood/profile_likelihood_results{tag}.csv"

    SCALE = "auto"  # read from <csv>_summary.json; or a float as in pool_paper
    # 'optimum': Delta relative to the optimum of the run; 'best': relative to the lowest cost
    # found, if profile points are below the optimum (negative Delta, refit did not converge)
    REFERENCE = "optimum"
    SHOW_TRUE_VALUE = False  # False: no green "true value" line (e.g. for real data or final figures)

    MAKE_GRID_PLOT = True
    GRID_OUT_PATH = path2 + f"profile_likelihood/profile_likelihood_grid{tag}.png"

    MAKE_INDIVIDUAL_PLOTS = True
    INDIVIDUAL_OUT_DIR = path2 + "profile_likelihood/profile_likelihood_individual"
    INDIVIDUAL_FILE_EXT = "pdf"
    # ================================================================

    _, result, true_params = load_calibration(path2, add_name)  # in-silico truth, if available
    # estimate for the plot if the run has not finished yet (otherwise read from the run files)
    param_opt = result["param_ode"]

    if MAKE_GRID_PLOT:
        plot_profile_likelihood_from_csv(CSV_PATH, confidence_level=0.95, scale=SCALE,
                                         save_path=GRID_OUT_PATH, ncols=4, true_params=true_params if SHOW_TRUE_VALUE else None,
                                         param_opt=param_opt, reference=REFERENCE)
    if MAKE_INDIVIDUAL_PLOTS:
        plot_profile_likelihood_individual(CSV_PATH, confidence_level=0.95, scale=SCALE,
                                           out_dir=INDIVIDUAL_OUT_DIR, file_ext=INDIVIDUAL_FILE_EXT,
                                           true_params=true_params if SHOW_TRUE_VALUE else None, param_opt=param_opt,
                                           reference=REFERENCE)

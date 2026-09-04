"""
Plot profile likelihood results from a saved CSV
==================================================

Standalone script: reads a CSV produced by run_profile_likelihood_all()
(columns: param_index, param_value, cost, <one column per parameter>)
and regenerates the profile-likelihood grid plot, without needing to
recompute anything.

Usage:
    python plot_profile_from_csv.py profile_likelihood_results_Hill.csv

Or import and call plot_profile_likelihood_from_csv() directly.
"""
import os
import sys

sys.path.append(os.getcwd())
import fusion_model as fm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
import string

colors_all = {
        'R': '#808080',
        'N_A':'#D06062',
        'N_B': '#4E89B1',
        'N':'#7E57A5',
        'T':'#99582A',
        'T_A':'#c79758',
        'N_lambd_1e-2_omega_0':'#E2B100',
        'N_lambd_1e-3_omega_0':'#386641',
        'N_lambd_1e-3_omega_0_5':'#0982A4',
        'N_wo_tempshift':'#679E48',
        'N_tempshift_10':'#ED733E',
        'N_tempshift_10_5_15':'#C3568A',
        'N_Lm': '#ED733E',
        'N_Lm_woT':'#D70040',
        'N_Ls23K': '#679E48',
        'N_Ls23Kco': '#386641',
        'N_Lm_withT':'#D06062',
        'N_LsCTC494': '#4E89B1',
        'N_LsCTC494co': '#00356B',  
    }

figsize_default = (6.5, 4.0)
figsize_default_small = (6.5, 2.0)
figsize_default2subpl = (13, 4.0)

plt.rcParams['figure.dpi'] = 400
plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

plt.rcParams['legend.fontsize'] = 14.
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.handlelength'] = 1.8
plt.rcParams['axes.prop_cycle'] = plt.cycler(linewidth=[2.5])
plt.rcParams['font.size'] = 15
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)


# Actual order of `param` as unpacked in the ODE function / as the CSV's
# parameter columns are laid out (this is the order param_index refers to):
#   mu_ls23K, mu_lsCTC494, mu_lm,
#   omega3, K3,
#   N_ls23K_texp, N_lsCTC494_texp, N_lm_texp,
#   kappa_T_0,
#   kappa_LA_ls23K_exp, kappa_LA_ls23K_2_exp,
#   kappa_LA_lsCTC494_exp, kappa_LA_lsCTC494_2_exp,
#   kappa_LA_lm_exp, kappa_LA_lm_2_exp
MODEL_PARAM_ORDER = [
    "mu_ls23K", "mu_lsCTC494", "mu_lm",
    "omega3", "K3",
    "N_t_ls23K", "N_t_lsCTC494", "N_t_lm",
    "kappa_T_0",
    "kappa_LA_ls23K", "kappa_LA_ls23K_2",
    "kappa_LA_lsCTC494", "kappa_LA_lsCTC494_2",
    "kappa_LA_lm", "kappa_LA_lm_2",
]

# Desired panel-letter order: growth rates -> N_t values -> omega/K -> kappa_T -> kappa_LA
PARAM_ORDER = [
    # Growth rates
    "mu_ls23K", "mu_lsCTC494", "mu_lm",
    # N_t values
    "N_t_ls23K", "N_t_lsCTC494", "N_t_lm",
    # Toxin effect: omega and K
    "omega3", "K3",
    # kappa_T
    "kappa_T_0",
    # kappa_LA values
    "kappa_LA_ls23K", "kappa_LA_ls23K_2",
    "kappa_LA_lsCTC494", "kappa_LA_lsCTC494_2",
    "kappa_LA_lm_2",
]

def get_panel_letter(param_index, model_param_order=MODEL_PARAM_ORDER, param_order=PARAM_ORDER):
    """Return the panel letter (A, B, C, ...) for a given parameter,
    identified by its position (param_index, as found in the CSV) in the
    original param array -- NOT by matching the CSV's LaTeX display name,
    which varies in formatting and can't be reliably matched by string
    equality against PARAM_ORDER.
    """
    try:
        key = model_param_order[param_index]
    except IndexError:
        raise ValueError(f"param_index {param_index} out of range for model_param_order.")
    try:
        letter_idx = param_order.index(key)
    except ValueError:
        raise ValueError(f"Key '{key}' (param_index={param_index}) not found in param_order list.")
    return string.ascii_uppercase[letter_idx]


def _add_Nt_axis(ax, name):
    """
    If `name` is one of the log10(N_t) columns (e.g. '$\\log_{10}{N^{Ls23K}_{t}}$'),
    attach a secondary top x-axis showing the physical N_t = 10**value.
    No-op for any other parameter name.
    """
    if "log_{10}" not in name.replace(" ", ""):
        return
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda v: 10.0 ** v, lambda v: np.log10(np.clip(v, 1e-300, None))),
    )
    secax.set_xlabel(r"$N_t$")
    secax.tick_params(labelsize=9)


def plot_profile_likelihood_from_csv(
    csv_path,
    confidence_level=0.95,
    scale="auto",
    save_path=None,
    ncols=4,
):
    """
    Read a profile-likelihood CSV and plot cost vs. parameter value for
    every profiled parameter, with the estimate marked, the chi2 threshold
    line, and the (approximate) confidence interval shaded.

    csv_path         : path to the CSV (param_index, param_value, cost, <param cols...>)
    confidence_level : for the chi2 threshold (default 0.95)
    scale            : "auto" estimates it from the data itself (see note below),
                        or pass a float to use a fixed scale directly.
    save_path        : if given, saves the figure to this path
    ncols            : number of subplot columns

    NOTE on scale="auto": since this script only has the CSV (not access to
    cost(), n_data, or sigma_hat2 from the original run), "auto" here just
    reuses whatever chi2 threshold made the original plot's shaded regions
    look right -- if you know the scale value printed during the original
    run (e.g. "Auto-estimated scale: ... scale=X"), pass it explicitly for
    an exact match instead of relying on the fallback below.
    """
    df = pd.read_csv(csv_path)

    free_idx = sorted(df["param_index"].unique())

    # The columns after param_index/param_value/cost are the parameter names,
    # in param_index order (one column per profiled parameter's own value at
    # each grid point). Use these as both display names and to recover the
    # point estimate (the row where param_value's cost is lowest for that index).
    param_name_cols = [c for c in df.columns if c not in ("param_index", "param_value", "cost")]

    # Recover param_opt (per parameter) and cost_opt (global minimum across all rows)
    param_opt = {}
    for idx in free_idx:
        sub = df[df["param_index"] == idx]
        best_row = sub.loc[sub["cost"].idxmin()]
        # the parameter's own column holds its value at that row
        col_name = param_name_cols[idx] if idx < len(param_name_cols) else None
        param_opt[idx] = best_row["param_value"]

    cost_opt = df["cost"].min()

    if scale == "auto":
        # Fallback heuristic: pick a small scale so the threshold sits just
        # above the typical in-grid cost variation, purely for a readable
        # plot when the true scale isn't available. Prefer passing the real
        # scale value explicitly (printed during the original run) instead.
        typical_spread = df.groupby("param_index")["cost"].apply(lambda s: s.max() - s.min()).median()
        scale = max(typical_spread / chi2.ppf(confidence_level, 1), 1e-12)
        print(f"[auto scale fallback] using scale={scale:.6g} (pass scale=<value> explicitly for an exact match)")

    threshold_chi2 = chi2.ppf(confidence_level, 1)  # fixed value (~3.84 for 95%, dof=1)

    n = len(free_idx)
    ncols_eff = min(ncols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols_eff))

    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(4.2 * ncols_eff, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, idx in enumerate(free_idx):
        ax = axes_flat[ax_i]
        sub = df[df["param_index"] == idx].sort_values("param_value")
        x = sub["param_value"].to_numpy()
        y_cost = sub["cost"].to_numpy()
        y = (y_cost - cost_opt) / scale  # rescaled onto the standard chi2 axis

        ax.plot(x, y, "o-", color=colors_all['N_LsCTC494co'], lw=1.5, ms=4)
        ax.axvline(param_opt[idx], color=colors_all['N_A'], ls="--", lw=1.2, label="estimate")

        y_lo, y_hi = np.min(y), np.max(y)
        pad = 0.1 * max(y_hi - y_lo, 1e-8)
        ax.set_ylim(max(y_lo - pad, -0.5), y_hi + pad)

        if threshold_chi2 <= y_hi + pad:
            ax.axhline(threshold_chi2, color="gray", ls=":", lw=1.2, label=f"{int(confidence_level*100)}\% threshold")
            below = np.where(y <= threshold_chi2)[0]
            if len(below):
                ax.axvspan(x[below[0]], x[below[-1]], color=colors_all['N_LsCTC494'], alpha=0.12)
        else:
            ax.text(0.98, 0.95, "threshold off-scale", transform=ax.transAxes,
                     ha="right", va="top", fontsize=7, color="gray")
        
        name = param_name_cols[idx] if idx < len(param_name_cols) else f"param[{idx}]"
        ax.set_xlabel(name, fontsize=11)
        #ax.set_xlabel("parameter value", fontsize=9)
        ax.set_ylabel(r"$\Delta(-2\log L)$ (approx.)", fontsize=9)
        ax.tick_params(labelsize=8)
        _add_Nt_axis(ax, name)

    for ax_j in range(n, len(axes_flat)):
        axes_flat[ax_j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_profile_likelihood_individual(
    csv_path,
    confidence_level=0.95,
    scale=None,
    out_dir=".",
    file_prefix="profile_",
    file_ext="png",
    figsize=(5.5, 4.0),
    coord_text=(0.12, 0.88),
    param_order=PARAM_ORDER,
    model_param_order=MODEL_PARAM_ORDER,
):
    """
    Same data/threshold logic as plot_profile_likelihood_from_csv(), but saves
    one separate figure per parameter instead of a single grid.

    out_dir      : directory to save the individual files into (created if missing)
    file_prefix  : prepended to each output filename
    file_ext     : "png", "pdf", etc.
    figsize      : size of each individual figure

    Filenames are built from the parameter's own column name in the CSV,
    sanitized to be filesystem-safe (LaTeX/special characters stripped),
    e.g. "$K_{inhib}$" -> "profile_K_inhib.png".

    Returns a dict {param_index: saved_file_path}.
    """
    import os
    import re

    if scale is None:
        raise ValueError(
            "scale not provided. Use the value printed as 'Auto-estimated scale: ... scale=X' "
            "during your original profile-likelihood run, or recompute via estimate_profile_scale()."
        )

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    free_idx = sorted(df["param_index"].unique())
    param_name_cols = [c for c in df.columns if c not in ("param_index", "param_value", "cost")]

    param_opt = {}
    for idx in free_idx:
        sub = df[df["param_index"] == idx]
        best_row = sub.loc[sub["cost"].idxmin()]
        param_opt[idx] = best_row["param_value"]

    cost_opt = df["cost"].min()
    threshold_chi2 = chi2.ppf(confidence_level, 1)  # fixed value (~3.84 for 95%, dof=1)

    def _sanitize(name):
        # strip $, \, {, }, ^ and similar LaTeX syntax, collapse to a safe filename fragment
        s = re.sub(r"[\$\\{}\^]", "", name)
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        return s

    saved = {}
    for idx in free_idx:
        sub = df[df["param_index"] == idx].sort_values("param_value")
        x = sub["param_value"].to_numpy()
        y_cost = sub["cost"].to_numpy()
        y = (y_cost - cost_opt) / scale  # rescaled onto the standard chi2 axis

        name = param_name_cols[idx] if idx < len(param_name_cols) else f"param_{idx}"
        

        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.plot(x, y, "o-", color=colors_all['N_LsCTC494co'], ms=6.5)
        ax.axvline(param_opt[idx], color=colors_all['N_A'], ls="--", label="estimate")

        y_lo, y_hi = np.min(y), np.max(y)
        pad = 0.1 * max(y_hi - y_lo, 1e-8)
        ax.set_ylim(max(y_lo - pad, -1), y_hi + pad)

        if threshold_chi2 <= y_hi + pad:
            ax.axhline(threshold_chi2, color="gray", lw=3, ls=":", label=f"{int(confidence_level*100)}\% threshold")
            below = np.where(y <= threshold_chi2)[0]
            if len(below):
                ax.axvspan(x[below[0]], x[below[-1]], color=colors_all['N_LsCTC494'], alpha=0.12)
        else:
            ax.text(0.98, 0.95, "threshold off-scale", transform=ax.transAxes,
                     ha="right", va="top", fontsize=8, color="gray")

        ax.set_xlabel(name, fontsize=16)
        #ax.set_xlabel("parameter value", fontsize=9)
        ax.set_ylabel(r"$\Delta(-2\log L)$")
        letter = get_panel_letter(idx, model_param_order, param_order)
        
        ax.text(*coord_text, rf'\textbf{{{letter}}}', transform=ax.transAxes)
        
        #ax.legend(frameon=False, loc='upper right')
        _add_Nt_axis(ax, name)
        fig.tight_layout()


        fname = f"Figures-pool_model_real_data_{file_prefix}{idx:02d}_{_sanitize(name)}.{file_ext}"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved[idx] = fpath
        print(f"Saved {fpath}")

    return saved


if __name__ == "__main__":
    # ================================================================
    # EDIT THESE, THEN JUST RUN THIS FILE (no command-line args needed)
    # ================================================================
    CSV_PATH = "pool_paper_casestudy/out/wo_pH_new/profile_likelihood_results_MM.csv"

    # The value printed as "Auto-estimated scale: ... scale=X" during
    # your original profile-likelihood run (see estimate_profile_scale()).
    SCALE = 0.000230382

    MAKE_GRID_PLOT = True
    GRID_OUT_PATH = "pool_paper_casestudy/out/wo_pH_new/profile_likelihood_grid.png"

    MAKE_INDIVIDUAL_PLOTS = True
    INDIVIDUAL_OUT_DIR = "pool_paper_casestudy/out/wo_pH_new/profile_likelihood_individual"
    INDIVIDUAL_FILE_EXT = "pdf"  # or "pdf"
    # ================================================================

    if MAKE_GRID_PLOT:
        plot_profile_likelihood_from_csv(
            CSV_PATH,
            confidence_level=0.95,
            scale=SCALE,
            save_path=GRID_OUT_PATH,
            ncols=4,
        )

    if MAKE_INDIVIDUAL_PLOTS:
        plot_profile_likelihood_individual(
            CSV_PATH,
            confidence_level=0.95,
            scale=SCALE,
            out_dir=INDIVIDUAL_OUT_DIR,
            file_ext=INDIVIDUAL_FILE_EXT,
        )

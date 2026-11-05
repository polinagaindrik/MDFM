"""Output folders of the model paper (paths relative to the repository root = working directory).

model_paper/out/
    model_complexity/   <n>_dim_<m>media_exp_<noise>noise/calibration/            (n_cl varied, 10 % noise)
    noise_vs_nspecies/  <noise>noise/<n>_dim_<m>media_exp_<noise>noise/calibration/  (n_cl x noise level)
    media_influence/    <media>_media/calibration/                                (media varied, 6 species)
    (run folders of designs other than PAPER_DESIGN = 'rep3' end in _<design>, runs with initial values
     other than 'uniform' in _x0<distribution>, e.g. ..._10noise_x0lognormal/; default X0_DISTRIBUTION = 'lognormal';
     multinomial MALDI with N colonies per plate in _maldi<N>, Dirichlet-multinomial NGS in _ngsA<A>, see NOISE_SCENARIO;
     fitted with the multinomial likelihood in _maldi<N>_mnfit, default MALDI_FIT = 'multinomial')
    test_2sp_<noise>noise/, test_2sp_<design>/                                    (2-species tests, test_2sp.py)

Usage:  path = study_path('model_complexity', n_cl=4, n_media=2, relnoise=0.1)
        name = add_name(n_cl=4, n_media=2)                    -> '_4dim_2media'
        path = study_path('media_influence', media='gen2+sel2'); name = add_name(6, media='gen2+sel2')
"""
import os

OUT_DIR = 'model_paper/out/'
STUDIES = ('model_complexity', 'noise_vs_nspecies', 'media_influence')
# experimental design of the paper runs (test_2sp.DESIGNS); other designs get the folder suffix _<design>
PAPER_DESIGN = 'rep3'
# distribution of the initial values x0 (setup/Initial_values_x0_paper_<distribution>.json, see
# fm.data.read_initial_values): default of the scripts; other distributions get the folder suffix _x0<distribution>
X0_DISTRIBUTION = 'lognormal'
# fit of multinomial MALDI data: 'multinomial' = multinomial likelihood (folder suffix _mnfit), 'clr' = log-ratio
# residuals of the relative noise model, undetected species dropped (no extra suffix)
MALDI_FIT = 'multinomial'
# fit of Dirichlet-multinomial NGS data: 'dm' = Dirichlet-multinomial likelihood (folder suffix _dmfit), 'clr' = log-ratio
# residuals of the relative noise model, undetected species dropped (no extra suffix)
NGS_FIT = 'dm'
# Noise of the in-silico data. All data generation, fit, profile and plot scripts follow NOISE_SCENARIO:
#   pc_sigma_log10: plate counts, Gaussian noise with this standard deviation on log10 of the counts
#                   (0.1: -21 % / +26 %, 0.15: -29 % / +41 %, 0.3: -50 % / +100 %; ZL2030 between subsamples: 0.15)
#   ngs_conc:       NGS, concentration A of the Dirichlet-multinomial with NGS_DEPTH reads per sample; relative error
#                   sqrt((1 - p) / p * (N + A) / (N (1 + A))): about 10 / 16 / 40 % at a share p = 0.02.
#                   From the ZL2030 replicate pairs (`14661` / `14661b`), on the species used by the ODE model (11 prevalent
#                   species + Others, get_bacteria_for_model(cutoff=0.1, cutoff_prop=0.001)):
#                     baseline 2000: pairs with >= 26000 reads (9 pairs, V15 and V16): A = 2590, 90 % interval 1671-4759;
#                                    only shares > 0.01: A = 1523 (1005-3285); about 70 % of the ZL2030 samples are this deep
#                     low 5000:      upper end of that interval
#                     high 300:      pairs with < 26000 reads (6 pairs, mostly V12): A = 233-374
#   maldi_n:        MALDI, multinomial with this number of identified colonies per plate (ZL2030: median 46, mode 48)
NOISE_SCENARIOS = {
    'low':      {'pc_sigma_log10': 0.1,  'ngs_conc': 5000., 'maldi_n': 96},
    'baseline': {'pc_sigma_log10': 0.15, 'ngs_conc': 2000., 'maldi_n': 48},
    'high':     {'pc_sigma_log10': 0.3,  'ngs_conc': 300.,  'maldi_n': 24},
}
NOISE_SCENARIO = 'baseline'  # 'low', 'baseline' or 'high'
NGS_DEPTH = 56000  # NGS reads per sample, in all scenarios (ZL2030: median 56330)


def noise_settings(scenario=None):
    """Noise of a scenario (None = NOISE_SCENARIO): {'pc_sigma_log10': sigma, 'ngs_dm': (N, A), 'maldi_n': N}."""
    scenario = NOISE_SCENARIO if scenario is None else scenario
    if scenario not in NOISE_SCENARIOS:
        raise ValueError(f'unknown noise scenario {scenario!r}, expected one of {list(NOISE_SCENARIOS)}')
    n = NOISE_SCENARIOS[scenario]
    return {'pc_sigma_log10': n['pc_sigma_log10'], 'ngs_dm': (NGS_DEPTH, n['ngs_conc']), 'maldi_n': n['maldi_n']}


PC_SIGMA_LOG10, NGS_DM, MALDI_N = (noise_settings()[k] for k in ('pc_sigma_log10', 'ngs_dm', 'maldi_n'))
_DEFAULT = object()

# media whose file suffix differs from the joined media names (first runs: gen1 -> '_gen', gen1+sel1 -> '')
_MEDIA_SUFFIX = {'gen1': 'gen', 'sel1': 'sel', 'gen1+sel1': '', 'gen0+sel1': ''}


def noise_tag(relnoise):
    """0.1 -> '10noise'."""
    return f'{int(round(relnoise * 100))}noise'


def x0_suffix(x0=None):
    """Folder suffix of the initial-value distribution x0 (None = X0_DISTRIBUTION): '' or e.g. '_x0lognormal'."""
    x0 = X0_DISTRIBUTION if x0 is None else x0
    return '' if x0 == 'uniform' else f'_x0{x0}'


def maldi_suffix(maldi_n=_DEFAULT, maldi_fit=None):
    """Folder suffix of the MALDI noise (default: MALDI_N) and its fit (None = MALDI_FIT): '' (Gaussian),
    e.g. '_maldi48' (multinomial data, log-ratio fit) or '_maldi48_mnfit' (multinomial data and likelihood)."""
    maldi_n = MALDI_N if maldi_n is _DEFAULT else maldi_n
    if maldi_n is None:
        return ''
    fit = MALDI_FIT if maldi_fit is None else maldi_fit
    if fit not in ('multinomial', 'clr'):
        raise ValueError(f"maldi_fit must be 'multinomial' or 'clr', got {fit!r}")
    return f'_maldi{int(maldi_n)}' + ('_mnfit' if fit == 'multinomial' else '')


def ngs_suffix(ngs_dm=_DEFAULT, ngs_fit=None):
    """Folder suffix of the NGS noise (default: NGS_DM) and its fit (None = NGS_FIT): '' (Gaussian), e.g.
    '_ngsA2000' (Dirichlet-multinomial data with A = 2000, log-ratio fit) or '_ngsA2000_dmfit' (data and likelihood)."""
    ngs_dm = NGS_DM if ngs_dm is _DEFAULT else ngs_dm
    if ngs_dm is None:
        return ''
    fit = NGS_FIT if ngs_fit is None else ngs_fit
    if fit not in ('dm', 'clr'):
        raise ValueError(f"ngs_fit must be 'dm' or 'clr', got {fit!r}")
    return f'_ngsA{int(ngs_dm[1])}' + ('_dmfit' if fit == 'dm' else '')


def study_dir(study):
    """Top folder of a study, e.g. 'model_paper/out/model_complexity/' (for summary plots of the study)."""
    if study not in STUDIES:
        raise ValueError(f'unknown study {study!r}, expected one of {STUDIES}')
    return OUT_DIR + study + '/'


def run_dir(study, n_cl=None, n_media=2, relnoise=PC_SIGMA_LOG10, media=None, design=None, x0=None, maldi_n=_DEFAULT,
            maldi_fit=None, ngs_dm=_DEFAULT, ngs_fit=None):
    """Folder of one run (contains calibration/, data_generation/, ...); design other than PAPER_DESIGN
    (None = PAPER_DESIGN): suffix _<design>, e.g. 4_dim_2media_exp_10noise_temps6/; initial values x0
    (None = X0_DISTRIBUTION) other than 'uniform': suffix _x0<x0>; multinomial MALDI (maldi_n, default MALDI_N):
    suffix _maldi<N>; Dirichlet-multinomial NGS (ngs_dm, default NGS_DM): suffix _ngsA<A>."""
    suffix = (f'_{design}' if design not in (None, PAPER_DESIGN) else '') + x0_suffix(x0) + maldi_suffix(maldi_n, maldi_fit)
    suffix += ngs_suffix(ngs_dm, ngs_fit)
    if study == 'model_complexity':
        return study_dir(study) + f'{int(n_cl)}_dim_{int(n_media)}media_exp_{noise_tag(relnoise)}{suffix}/'
    if study == 'noise_vs_nspecies':
        return (study_dir(study) + f'{noise_tag(relnoise)}/'
                f'{int(n_cl)}_dim_{int(n_media)}media_exp_{noise_tag(relnoise)}{suffix}/')
    if study == 'media_influence':
        if media is None:
            raise ValueError("study 'media_influence' needs media, e.g. 'gen0' or 'gen2+sel2'")
        return study_dir(study) + f'{media}_media{suffix}/'
    raise ValueError(f'unknown study {study!r}, expected one of {STUDIES}')


def study_path(study, n_cl=None, n_media=2, relnoise=PC_SIGMA_LOG10, media=None, sub='calibration', design=None, x0=None,
               maldi_n=_DEFAULT, maldi_fit=None, ngs_dm=_DEFAULT, ngs_fit=None):
    """Calibration folder of one run (sub='' for the run folder itself, e.g. sub='data_generation')."""
    path = run_dir(study, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=media, design=design, x0=x0,
                   maldi_n=maldi_n, maldi_fit=maldi_fit, ngs_dm=ngs_dm, ngs_fit=ngs_fit)
    return path + sub + '/' if sub else path


def media_n(media):
    """'gen2+sel2' -> 2."""
    return len(media.split('+'))


def add_name(n_cl, n_media=None, media=None):
    """File suffix of a run: '_4dim_2media'; with media (media_influence) e.g. '_6dim_2media_gen2sel2'."""
    if media is None:
        return f'_{int(n_cl)}dim_{int(n_media)}media'
    suffix = _MEDIA_SUFFIX.get(media, media.replace('+', ''))
    return f'_{int(n_cl)}dim_{media_n(media)}media' + (f'_{suffix}' if suffix else '')


def generator_name(media):
    """Name of the in-silico data generator in fusion_model.data for a media combination."""
    if media_n(media) == 1:
        return f'model_1media_exp_{media}'
    return 'model_2media_exp' + ('gensel' if media == 'gen0+sel1' else media.replace('+', ''))


def test_2sp_path(design='base', relnoise=PC_SIGMA_LOG10, normalize_true_s=False, x0=None, maldi_n=_DEFAULT, maldi_fit=None,
                  ngs_dm=_DEFAULT, ngs_fit=None):
    """Folder of test_2sp.py: base design -> test_2sp_<noise>noise/, other designs -> test_2sp_<design>/
    (10 % noise) or test_2sp_<design>_<noise>noise/; initial values x0 (None = X0_DISTRIBUTION) other than
    'uniform': suffix _x0<x0>."""
    if design == 'base':
        path = OUT_DIR + f'test_2sp_{noise_tag(relnoise)}/'
    else:
        path = OUT_DIR + f'test_2sp_{design}/'
        if relnoise != 0.1:
            path = path.rstrip('/') + f'_{noise_tag(relnoise)}/'
    if normalize_true_s:
        path = path.rstrip('/') + '_normS/'
    return path.rstrip('/') + x0_suffix(x0) + maldi_suffix(maldi_n, maldi_fit) + ngs_suffix(ngs_dm, ngs_fit) + '/'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

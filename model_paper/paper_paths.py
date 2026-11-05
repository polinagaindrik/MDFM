"""Output folders of the model paper (paths relative to the repository root = working directory).

model_paper/out/
    model_complexity/   <n>_dim_<m>media_exp_<noise>noise/calibration/            (n_cl varied, 10 % noise)
    noise_vs_nspecies/  <noise>noise/<n>_dim_<m>media_exp_<noise>noise/calibration/  (n_cl x noise level)
    media_influence/    <media>_media/calibration/                                (media varied, 6 species)
    (run folders of designs other than PAPER_DESIGN = 'rep3' end in _<design>, runs with initial values
     other than 'uniform' in _x0<distribution>, e.g. ..._10noise_x0lognormal/; default X0_DISTRIBUTION = 'lognormal';
     multinomial MALDI with N colonies per plate in _maldi<N>, default MALDI_N = 48 as in the ZL2030 data)
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
# MALDI noise of the in-silico data: None = relative Gaussian noise, N = multinomial with N identified colonies per
# plate (fm.data.prepare_insilico_data(maldi_n=N)); default of the scripts, N gets the folder suffix _maldi<N>
MALDI_N = 48
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


def maldi_suffix(maldi_n=_DEFAULT):
    """Folder suffix of the MALDI noise (default: MALDI_N): '' (Gaussian) or e.g. '_maldi24' (multinomial)."""
    maldi_n = MALDI_N if maldi_n is _DEFAULT else maldi_n
    return '' if maldi_n is None else f'_maldi{int(maldi_n)}'


def study_dir(study):
    """Top folder of a study, e.g. 'model_paper/out/model_complexity/' (for summary plots of the study)."""
    if study not in STUDIES:
        raise ValueError(f'unknown study {study!r}, expected one of {STUDIES}')
    return OUT_DIR + study + '/'


def run_dir(study, n_cl=None, n_media=2, relnoise=0.1, media=None, design=None, x0=None, maldi_n=_DEFAULT):
    """Folder of one run (contains calibration/, data_generation/, ...); design other than PAPER_DESIGN
    (None = PAPER_DESIGN): suffix _<design>, e.g. 4_dim_2media_exp_10noise_temps6/; initial values x0
    (None = X0_DISTRIBUTION) other than 'uniform': suffix _x0<x0>; multinomial MALDI (maldi_n, default MALDI_N):
    suffix _maldi<N>."""
    suffix = (f'_{design}' if design not in (None, PAPER_DESIGN) else '') + x0_suffix(x0) + maldi_suffix(maldi_n)
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


def study_path(study, n_cl=None, n_media=2, relnoise=0.1, media=None, sub='calibration', design=None, x0=None,
               maldi_n=_DEFAULT):
    """Calibration folder of one run (sub='' for the run folder itself, e.g. sub='data_generation')."""
    path = run_dir(study, n_cl=n_cl, n_media=n_media, relnoise=relnoise, media=media, design=design, x0=x0,
                   maldi_n=maldi_n)
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


def test_2sp_path(design='base', relnoise=0.1, normalize_true_s=False, x0=None, maldi_n=_DEFAULT):
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
    return path.rstrip('/') + x0_suffix(x0) + maldi_suffix(maldi_n) + '/'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

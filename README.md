<a name="readme-top"></a>

<h3 align="center">MDFM: Microbiome Data Fusion Model</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#theoretical-background">Theoretical Background</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#quick-start">Quick Start</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#prepare-data">Prepare Data</a></li>
        <li><a href="#model-calibration">Model Calibration</a></li>
        <li><a href="#model-prediction">Model Prediction</a></li>
        <li><a href="#local-refinement-and-identifiability">Local Refinement and Identifiability</a></li>
      </ul>
    </li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

MDFM (Python package `fusion_model`) is a temperature-dependent model of bacterial growth in food that fuses three kinds of microbiome measurements:

- bacterial plate counts on different media (microbiology),
- species proportions from MALDI-ToF,
- species proportions from amplicon sequencing (NGS).

The package calibrates the model on these data, estimating the growth parameters of each species, the interactions between species and the recovery of each species on each medium. With the calibrated model it then predicts the bacterial dynamics of new experiments from their storage temperature and initial values, and analyses which parameters the data can identify.

It was developed for the project [Zukunftslabor2030 (ZL2030)](https://zukunftslabor2030.de), which aims at the dynamic assessment and prediction of food spoilage. This repository is the core library: the model, cost functions, optimization and plotting, with examples on generated in-silico data. It contains no measurement data and no application to a specific data set.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Theoretical Background

The mathematical model will be described in a forthcoming paper. In short, the main model `fusion_model2` is a system of ODEs for the bacterial count of each species with a lag phase, temperature-dependent growth rates, a common carrying capacity and pairwise inhibition between species. Each data type observes these counts differently:

- plate counts and MALDI-ToF see the species filtered by a media matrix `S` (media x species),
- NGS sees the proportions of the species, weighted by `T_x`.

The parameter vector of a calibration with `fm.pest.cost_withS` and `fusion_model2` is:

```
[ x0 (n_cl per experiment, log10) | lambda_1 (n_cl) | lambda_exp (n_cl) | mu_0 (n_cl) | mu_exp (n_cl) | N_1 | N_exp | k_ij (n_cl x n_cl) | S (n_media x n_cl) ]
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- Python 3.11 or newer
- A LaTeX installation with `dvipng` for the figures (the plotting functions use `text.usetex`)

### Installation

Clone the repository and install the package, ideally in a virtual environment:

```sh
git clone https://github.com/polinagaindrik/MDFM
cd MDFM
pip install -e .
```

Two optional extras speed up the calibration:

```sh
pip install -e ".[numba]"    # about 6x faster ODE solves of fusion_model2
pip install -e ".[jax]"      # exact derivatives for the local fits (backend='jax' of fm.pest.local_optimization)
```

For a GPU, install JAX with `pip install "jax[cuda12]"`. JAX then uses the GPU by default. With many parallel processes, for example in `fm.pest.multistart_optimization`, large problems can run out of GPU memory; set `JAX_PLATFORMS=cpu` to use one CPU core per process instead. Note that `requirements.txt` installs the dependencies including both extras, but not the package itself.

### Quick Start

[`main.py`](main.py) is a complete example on in-silico data that needs no measurement data or setup files. It generates data for 2 species on 2 media, calibrates the model, predicts a new experiment and writes the results and figures to `out/main_example/`:

```sh
python main.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

The steps below follow [`main.py`](main.py), which shows them in context. All snippets use:

```python
import fusion_model as fm
```

The main submodules are `fm.data` (data), `fm.mdl` (ODE models), `fm.pest` (parameter estimation), `fm.plotting` and `fm.output` (JSON results).

### Prepare Data

All functions return the data frames `dfs = [df_platecount, df_maldi, df_ngs]`, the species names, the NGS weights `T_x` and the predefined entries of the media matrix `s_x_predefined` (`nan` = to be estimated).

**In-silico data.** A generator such as `fm.data.model_2media_expgensel` simulates `fusion_model2` with given parameters:

```python
S_matrix_setup = {'s_selective': [[0.87, 0.09]], 's_general0': [0.7, 0.56], 'T_x': [1., 1.]}
dfs, bact, T_x, s_x_predefined = fm.data.prepare_insilico_data(
    fm.data.model_2media_expgensel, n_cl, temps, ntr, S_matrix_setup,
    param_ode=param_ode_true, x10=x10, path=path, inhib=True, rel_noise=0.1)
```

Here `n_cl` is the number of species, `temps` the temperature of each experiment, `ntr` the number of trajectories per temperature and `x10` the log10 initial values (experiments x species). The folder `path` has to exist. Instead of the relative Gaussian noise `rel_noise`, each data type can have a noise model of its own: `platecount_log10_sd` (Gaussian on log10 of the counts), `maldi_n` (multinomial) and `ngs_dm` (Dirichlet-multinomial).

**Measured data.** Provide the same three data frames from your own measurements:

| Data frame | Rows | Columns, e.g. experiment `V01`, atmosphere `M1`, 2 °C, day 3, medium `PC` |
|---|---|---|
| `df_platecount` | `Average`, `Replicas`, `Standard deviation` | `V01_M1_02C_03_PC-platecount` |
| `df_maldi` | Species, proportions | `V01_M1_02C_03_PC-maldi` |
| `df_ngs` | Species, proportions | `V01_M1_02C_03_ngs` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Calibration

Define the calibration setup. `define_calibr_setup_insilico` adds the parameter bounds, `lambda_exp_max` sets the upper bound of `lambda_exp`:

```python
calibr_setup = fm.pest.define_calibr_setup_insilico({
    'model': fm.mdl.fusion_model2,
    'T_x': T_x,
    'workers': -1,                     # processes of differential evolution, -1 = all cores
    'output_path': path,
    'n_cl': n_cl,
    'n_media': n_media,
    'media': media,                    # sorted media names, e.g. ['gen0', 'sel1']
    'dfs': dfs,
    'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
    'exps': exps,                      # e.g. ['V01', 'V02', 'V03']
    'exp_temps': exp_temps,            # e.g. {'V01': 2., 'V02': 6., 'V03': 10.}
}, inhib=True, s_x_predefined=s_x_predefined)
```

Run the parameter estimation (differential evolution). The history of the optimization is written to `fm.pest.optimization.output_file`, by default in `out/`, so this folder has to exist or the file has to be set to another path:

```python
param_opt, cost_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)
```

Split the result and plot the fit for each experiment:

```python
param_ode = param_opt[n_cl*len(exps):-n_cl*n_media]
s_x = param_opt[-n_cl*n_media:].reshape((n_media, n_cl))
fm.plotting.plot_optimization_result(param_opt[:-n_cl*n_media], dict(calibr_setup, s_x=s_x), t_plot, path=path)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Prediction

The ODE parameters and `S` stay fixed from the calibration, and only the initial values of the new experiments are estimated:

```python
prediction_setup = dict(calibr_setup,
                        param_ode=param_ode,
                        s_x=s_x,
                        dfs=dfs_predict,
                        param_bnds=x0_bnds,    # bounds of the log10 initial values, n_cl per experiment
                        exps=exps_predict,
                        exp_temps=exp_temps_predict)
x0_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals, prediction_setup)[0]
```

Plot the prediction. `x0_dict` maps each experiment to its initial values:

```python
x0_dict = {exp: x0_opt[n_cl*i:n_cl*(i+1)] for i, exp in enumerate(exps_predict)}
fm.plotting.plot_prediction_result(param_ode, x0_dict, prediction_setup, t_plot, path=path)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Local Refinement and Identifiability

Differential evolution usually ends close to, but not at, a local minimum. To refine the result (methods `'L-BFGS-B'` or `'least_squares'`):

```python
param_opt, cost_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup, local_polish=True,
                                                     local_options={'method': 'least_squares'})
```

`fm.pest.multistart_optimization` is an alternative global search with many local fits from Latin hypercube start points.

`fm.pest.profile_likelihood` computes profile likelihoods with confidence intervals of the parameters and a classification of their identifiability (`run_profile_likelihood_all`, `run_profile_likelihood_adaptive`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Repository Structure

| Path | Content |
|---|---|
| `fusion_model/data` | In-silico data generation, reading and writing of results |
| `fusion_model/model` | ODE models and observables |
| `fusion_model/parameter_estimation` | Cost functions and likelihoods, global and local optimization, media matrix, profile likelihood, JAX backend |
| `fusion_model/plotting` | Figures of data, fits and parameters |
| `fusion_model/dimensionality_reduction` | Projection and clustering of species |
| `fusion_model/analysis` | Spectral expansion |
| `fusion_model/tools` | Data frame helpers |
| `main.py` | Complete example on in-silico data |
| `experiments/insilico/` | Example of generated in-silico data frames |
| `examples/` | Parameter distribution studies on in-silico data, plots of saved results |
| `others/` | Notes and earlier scripts, not maintained |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the MIT License, see [`LICENSE`](LICENSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="readme-top"></a>

<h3 align="center">MDFM: Microbiome Data Fusion Model</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
        <li>
      <a href="#theoretical-background">Theoretical Background</a>
      <ul>
        <li><a href="#data-fusion-model">Data Fusion Model</a></li>
      </ul>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <ul>
        <li><a href="#download-data">Download Data</a></li>
        <li><a href="#model-calibration">Model Calibration</a></li>
        <li><a href="#model-prediction">Model Prediction</a></li>
    </ul>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

`fusion_model` is a tool developed for the project [Zukunftslabor2030 (ZL2030)](https://zukunftslabor2030.de) that aims the dynamic assessment and prediction of food spoilage on the basis of different measurement data.
This tool focuses on developing a temperature-dependent data fusion shelf-life model by analyzing bacterial count data (MiBi) and bacterial diversity data (NGS, MALDI-ToF).
By calibrating the model with microbiological measurements, it predicts the shelf life of the product based on input storage temperature conditions and an initial number of bacteria.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Theoretical Background

### Data Fusion Model
The mathematical model developed for data fusion of the microbiome data could be soon found in the coming out paper.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

For this tool a working installation of the python scripting language version `python 3.11` is required. All necessary libraries could be found in `requirements.txt` and can be installed thin `pip`:
```sh
  pip install -r requirements.txt
```

### Installation

To get started with the `fusion_model` project the user is expected to clone the git repository:
```sh
   git clone https://github.com/polinagaindrik/fusion_model
```
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

To start with the Digital Twin project we use a `main.py` file that allows us to download data, calibrate the model, and predict the behavior of the model for input temperature conditions. Below is a brief overview of the primary functions available.

At the beginning we import the functions of the source code to our main file:
```sh
  import os
  import sys
  sys.path.append(os.getcwd())
  import DT_diversity as fm
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Download Data

The Excel sheets containing microbiology (MiBi) experimental data stored
in OpenBis can be easily downloaded with the function:
```sh
fm.data.download_microbiology_from_openbis(path, username, password)
```
Here `path="/../"` defines the location where Excel files are saved.
`username`, `password` are personal login data for OpenBis.

  
MALDI-ToF data:
```sh
fm.data.download_MALDI_from_openbis(path, username, password)
```
NGS data:
```sh
fm.data.download_NGS_from_openbis(path, username, password)
```

To prepare the ZL2030 data for model calibration use the function:
```sh
dfs_calibr, bact, T, s_predef = fm.data.prepare_ZL2030_data(exps_mibi, exps_maldi, exps_ngs, cutoff=0.01)
```
where `exps_mibi`, `exps_maldi`, `exps_ngs` define available experiments for each data type that we want to use for model calibration.

Or another way to get the data is to generate the in-silico data for a defined insilico_model (e.g. from fm.data.):
```sh
dfs_calibr, bact, T, s_predef =  fm.data.prepare_insilico_data(insilico_model, temps, ntr)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Model Calibration

At the beginning we need to define calibration setup, for example:
```sh
calibr_setup={
        'model': ode_model,
        'param_bnds':p_bnds,
        'T_x': T,
        'workers': -1,
        'output_path': path,
        'n_cl': n_cl,
        'n_media': n_media,
        'dfs': dfs_calibr,
        'aggregation_func': fm.pest.cost_sum_and_geometric_mean,
        'exps': exps_mibi,
        'exp_temps': ['V01':2.,]
        }
```

To run the parameter estimation function:
```sh
param_opt = fm.pest.calculate_model_params(fm.pest.cost_withS, calibr_setup)[0]
```
To plot the estimated model for each experiment:
```sh
fm.plotting.plot_optimization_result(param_opt, calibr_setup, time_array)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Prediction

The initial conditions for each experiment are different and needed to be estimated for correct model prediction. For this we define prediction setup with `param_ode` and `s_x` estimated in `param_opt` and `dfs_predict`, `exps_predict` defining experiments for prediction:
```sh
    prediction_setup = calibr_setup
    prediction_setup['param_ode'] = param_ode
    prediction_setup['dfs'] = dfs_predict
    prediction_setup['param_bnds'] = x0_bnds
    prediction_setup['exps'] = exps_predict
    prediction_setup['s_x'] = s_x
```
To run the initial values estimation for the model prediction:
```sh
x0_opt = fm.pest.calculate_prediction(fm.pest.cost_initvals, prediction_setup)[0]
```
To plot the estimated model for each experiment:
```sh
fm.plotting.plot_prediction_result(prediction_setup['param_ode'], x0_opt, prediction_setup, time_array)
```

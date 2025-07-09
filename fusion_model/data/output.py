#!/usr/bin/env python3

import numpy as np
import json
from pydantic.json import custom_pydantic_encoder

# TODO Create ouput file for model calculations

# The function from (FisInMa, thanks to Jonas)
def _get_encoder(calibr_result):
    encoders = {
        np.ndarray: lambda x: x.tolist(),
        np.int32: lambda x: str(x),
        np.int64: lambda x: "here",
    }
    # Define the encoder as a modification of the pydantic encoder
    return lambda obj: custom_pydantic_encoder(encoders, obj)


# Save the modell calibr_result to the json file
def json_dump(calibr_result, filename, dir='', **kwargs):
    # Special encoders for any object we might come across
    if "default" not in kwargs.keys():
        kwargs["default"] = _get_encoder(calibr_result)
    if "indent" not in kwargs.keys():
        kwargs["indent"] = 4

    # Return the json output as stri
    with open(dir+filename, "w") as f:
        json.dump(calibr_result, f, **kwargs)

# TODO convert read solution to the format it was before saving
def read_from_json(filename, dir=''):
    f = open(dir + filename,)
    sol = json.load(f)
    return sol


def save_values_each_experiment(vals_opt, exps, n_cl, dir='', filename=''):
    vals_dict = {}
    for i, exp in enumerate(exps):
        vals_dict[exp] = vals_opt[n_cl*i:n_cl*(i+1)]
    json_dump(vals_dict, f'{filename}.json', dir=dir)
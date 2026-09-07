"""
Ported from ``fusion_model/data/output.py``.

Handles saving/loading calibration results (and any other JSON-serializable
result dict) to/from disk. Kept byte-for-byte behaviourally identical to the
original so that JSON files produced/read by older runs stay compatible.
"""

import json

import numpy as np
from pydantic.json import custom_pydantic_encoder


def _get_encoder(calibr_result):
    encoders = {
        np.ndarray: lambda x: x.tolist(),
        np.int32: lambda x: str(x),
        np.int64: lambda x: "here",
    }
    # Define the encoder as a modification of the pydantic encoder
    return lambda obj: custom_pydantic_encoder(encoders, obj)


def json_dump(calibr_result, filename, dir='', **kwargs):
    """Save `calibr_result` (or any JSON-serializable object) to a json file."""
    # Special encoders for any object we might come across
    if "default" not in kwargs.keys():
        kwargs["default"] = _get_encoder(calibr_result)
    if "indent" not in kwargs.keys():
        kwargs["indent"] = 4

    with open(dir + filename, "w") as f:
        json.dump(calibr_result, f, **kwargs)


def read_from_json(filename, dir=''):
    """Load a previously json_dump-ed object back from disk."""
    with open(dir + filename) as f:
        sol = json.load(f)
    return sol


def save_values_each_experiment(vals_opt, exps, n_cl, dir='', filename=''):
    vals_dict = {}
    for i, exp in enumerate(exps):
        vals_dict[exp] = vals_opt[n_cl * i:n_cl * (i + 1)]
    json_dump(vals_dict, f'{filename}.json', dir=dir)

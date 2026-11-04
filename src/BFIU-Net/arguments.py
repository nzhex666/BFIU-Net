from __future__ import annotations

import argparse
from collections.abc import Mapping
import math
import os
from pathlib import Path


COMMON_DEFAULTS = {
    "model": "BFIU_Net", "patch_size": (128, 128, 128), "batch_size": 1,
    "normalization": "zscore", "loss": "bce", "threshold": 0.5,
}
TRAIN_DEFAULTS = {
    "epochs": 40, "learning_rate": 0.0001, "seed": 42, "shuffle": True,
    "train_manifest": "data/train.json", "validation_manifest": "data/validation.json",
}


def _model_name(value):
    from .models import normalize_model_name
    return normalize_model_name(value)


def add_common_arguments(parser):

    from .models import list_models
    parser.add_argument("--model", type=_model_name, choices=list_models(), default="BFIU_Net")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128),
                        metavar=("D", "H", "W"), help="Fixed patch dimensions, each divisible by 4 (default: 128 128 128)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--normalization", choices=("zscore", "none"), default="zscore",
                        help="Per-input-patch preprocessing (default: zscore)")
    parser.add_argument("--loss", choices=("bce", "balanced_bce"), default="bce")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Inclusive probability threshold in [0,1] (default: 0.5)")


def _path(value, name):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(name + " must be a nonempty path")
    try:
        return str(Path(value).expanduser().resolve())
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ValueError("Invalid or inaccessible " + name + " path") from exc


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def _finite_number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(name + " must be a finite number")


def _validate(options, *, training):
    if isinstance(options, argparse.Namespace):
        options = vars(options)
    if not isinstance(options, Mapping):
        raise ValueError("Parameters must be a dictionary or argparse.Namespace, not a configuration-file path")
    required = ("output_dir",) if training else ("weights", "output")
    paths = (("train_manifest", "validation_manifest", "output_dir") if training
             else ("manifest", "weights", "output"))
    defaults = dict(COMMON_DEFAULTS)
    if training:
        defaults.update(TRAIN_DEFAULTS)
    else:
        defaults["manifest"] = "data/test.json"
    unknown = set(options) - set(defaults) - set(required)
    if unknown:
        raise ValueError("Unknown parameter(s): " + ", ".join(sorted(map(str, unknown))))
    missing = [name for name in required if name not in options]
    if missing:
        raise ValueError("Missing required parameter(s): " + ", ".join(missing))
    values = dict(defaults)
    values.update(options)
    values["model"] = _model_name(values["model"])
    shape = values.pop("patch_size")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(v) is not int or v < 4 or v % 4 for v in shape)):
        raise ValueError("patch_size must contain three positive integer dimensions divisible by 4")
    values["input_shape"] = list(shape) + [1]
    _positive_integer(values["batch_size"], "batch_size")
    if values["normalization"] not in ("zscore", "none"):
        raise ValueError("normalization must be zscore or none")
    values["preprocessing"] = values.pop("normalization")
    if values["loss"] not in ("bce", "balanced_bce"):
        raise ValueError("loss must be bce or balanced_bce")
    _finite_number(values["threshold"], "threshold")
    if not 0 <= values["threshold"] <= 1:
        raise ValueError("threshold must be in [0,1]")
    if training:
        _positive_integer(values["epochs"], "epochs")
        _finite_number(values["learning_rate"], "learning_rate")
        if values["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")
        if type(values["seed"]) is not int or not 0 <= values["seed"] < 2 ** 32:
            raise ValueError("seed must be an integer in [0,2**32)")
        if type(values["shuffle"]) is not bool:
            raise ValueError("shuffle must be a boolean")
        values["optimizer"] = "adam"
    for name in paths:
        values[name] = _path(values[name], name)
    return values


def validate_training_arguments(options):

    return _validate(options, training=True)


def validate_evaluation_arguments(options):

    return _validate(options, training=False)

"""YAML configuration loading with single-level inheritance and validation."""

from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def load(path):
    """Load a config, resolving `extends: [paths]` relative to the file itself."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    bases = raw.pop("extends", [])
    result = {}
    for base in [bases] if isinstance(bases, str) else bases:
        result = merge(result, load(path.parent / base))
    return merge(result, raw)


def set_nested(config, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


def validate(c):
    from dfkd.augment import Augment
    from dfkd.data import DATASETS
    from dfkd.models import MODELS
    from dfkd.synthetic import SyntheticDataset

    d = c["dataset"]
    spec = DATASETS.resolve(d["name"])
    for field in ("channels", "num_classes"):
        if d[field] != spec[field]:
            raise ValueError(f"dataset.{field}: expected {spec[field]}, got {d[field]}")
    norm = d["normalization"]
    if len(norm["mean"]) != d["channels"] or len(norm["std"]) != d["channels"]:
        raise ValueError("Normalization needs one mean and std per channel")
    if min(norm["std"]) <= 0:
        raise ValueError("Normalization std must be positive")
    for role in ("teacher", "student"):
        MODELS.resolve(c[role]["architecture"])
    if c["distillation"]["temperature"] <= 0:
        raise ValueError("distillation.temperature must be positive")
    if c["training"]["batch_size"] < 1 or c["training"]["epochs"] < 1:
        raise ValueError("training.batch_size and training.epochs must be positive")
    Augment(c["augmentation"], d)
    SyntheticDataset(c)
    return c

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
    from dfkd.schedule import CURVES
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
    t = c["training"]
    schedule = t["batch_size_schedule"]
    if not schedule or min(schedule) < 1:
        raise ValueError("training.batch_size_schedule needs at least one positive batch size")
    if t["epochs"] < len(schedule):
        raise ValueError("training.epochs must give every batch-size phase at least one epoch")
    samples = c["synthetic_data"]["num_samples"]
    if any(min(samples, b) < 2 or samples % b == 1 for b in schedule):
        # A singleton final minibatch makes BatchNorm fail in training mode.
        raise ValueError(
            "batch_size_schedule leaves a single-sample final minibatch; "
            "adjust synthetic_data.num_samples or the schedule"
        )
    for spec in (c["scheduler"], c["teacher_training"]["scheduler"]):
        CURVES.resolve(spec["name"])
        if not 0 <= spec.get("min_lr_ratio", 0) <= 1:
            raise ValueError("scheduler.min_lr_ratio must be in [0, 1]")
    Augment(c["augmentation"], d)
    SyntheticDataset(c)
    return c

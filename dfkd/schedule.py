"""Batch size and LR scheduling shared by teacher and student training.

Student defaults use fixed 50-epoch phases from the reference notebook. With
phase_epochs omitted/None, divide the epoch budget evenly (teacher/legacy mode).
Cosine endpoints are inclusive, and the LR restarts at each phase boundary.
"""

import math

from dfkd.registry import Registry

CURVES = Registry()


@CURVES.register("constant")
def constant(t, n, config):
    return 1.0


@CURVES.register("cosine")
def cosine(t, n, config):
    floor = config.get("min_lr_ratio", 0.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / max(1, n - 1)))


@CURVES.register("step")
def step(t, n, config):
    return config.get("gamma", 0.1) ** (t // config.get("step_size", 30))


@CURVES.register("warmup_cosine")
def warmup_cosine(t, n, config):
    warmup = config.get("warmup_epochs", 5)
    if t < warmup:
        return (t + 1) / max(1, warmup)
    return cosine(t - warmup, max(1, n - warmup), config)


def phase_at(epoch, epochs, sizes, phase_epochs=None):
    """Return (phase index, phase start epoch, phase length, batch size)."""
    if not 0 <= epoch < epochs:
        raise ValueError(f"Epoch {epoch} lies outside the configured {epochs}-epoch schedule")
    if phase_epochs is not None:
        phase = min(epoch // phase_epochs, len(sizes) - 1)
        start = phase * phase_epochs
        # The final BS persists if training is extended; cosine restarts every phase_epochs.
        return phase, start, phase_epochs, sizes[phase]
    quotient, remainder = divmod(epochs, len(sizes))
    start = 0
    for i, batch_size in enumerate(sizes):
        length = quotient + (i < remainder)
        if start <= epoch < start + length:
            return i, start, length, batch_size
        start += length
    raise ValueError(f"Epoch {epoch} lies outside the configured {epochs}-epoch schedule")


class Schedule:
    """Drives the learning rate; reports the batch size for the current epoch."""

    def __init__(self, optimizer, config, training):
        self.optimizer = optimizer
        self.config = config
        self.epochs = training["epochs"]
        self.sizes = training["batch_size_schedule"]
        self.phase_epochs = training.get("phase_epochs")
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        CURVES.resolve(config["name"])

    def step(self, epoch):
        """Set the LR for `epoch` and return that epoch's (phase, batch size)."""
        phase, start, length, batch_size = phase_at(epoch, self.epochs, self.sizes, self.phase_epochs)
        if self.config.get("phase_resets", False):
            t, n = (epoch - start) % length, length
        else:
            t, n = epoch, self.epochs
        factor = CURVES.resolve(self.config["name"])(t, n, self.config)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor
        return phase, batch_size

"""Random synthetic image priors and the virtual corpus they define.

No generator network and no real images: every sample is drawn deterministically
from its index, so the corpus is reproducible and costs O(1) memory.
"""

import math

import torch
from torch.utils.data import Dataset

from dfkd.registry import Registry

PRIORS = Registry()


def _grid(size):
    v = torch.linspace(-1, 1, size)
    return torch.meshgrid(v, v, indexing="ij")


def _rand(g, low=0.0, high=1.0):
    return low + (high - low) * torch.rand((), generator=g).item()


def _colorize(field, channels, g):
    """Map a scalar field to a random per-channel colour range in [0, 1]."""
    low = torch.rand(channels, 1, 1, generator=g)
    high = torch.rand(channels, 1, 1, generator=g)
    return low + (high - low) * field[None]


def _unit(field):
    return (field - field.min()) / (field.max() - field.min()).clamp_min(1e-8)


@PRIORS.register("uniform")
def uniform(channels, size, g):
    return torch.rand(channels, size, size, generator=g)


@PRIORS.register("gaussian")
def gaussian(channels, size, g, mean=0.5, std=0.2):
    if std < 0:
        raise ValueError("gaussian std must be nonnegative")
    return (torch.randn(channels, size, size, generator=g) * std + mean).clamp(0, 1)


@PRIORS.register("linear_gradient")
def linear_gradient(channels, size, g):
    y, x = _grid(size)
    angle = _rand(g, 0, 2 * math.pi)
    return _colorize(_unit(x * math.cos(angle) + y * math.sin(angle)), channels, g)


@PRIORS.register("checkerboard")
def checkerboard(channels, size, g, block_sizes=(2, 4, 8, 16)):
    choices = [b for b in block_sizes if 1 <= b <= size]
    if not choices:
        raise ValueError("No checkerboard block size fits the image")
    b = choices[torch.randint(len(choices), (), generator=g).item()]
    phase = torch.randint(2 * b, (2,), generator=g)
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return _colorize((((y + phase[0]) // b + (x + phase[1]) // b) % 2).float(), channels, g)


@PRIORS.register("gabor")
def gabor(channels, size, g, frequency=(1.0, 8.0), sigma=(0.15, 0.8)):
    y, x = _grid(size)
    a = _rand(g, 0, 2 * math.pi)
    u = x * math.cos(a) + y * math.sin(a)
    width = _rand(g, *sigma)
    if width <= 0:
        raise ValueError("gabor sigma must be positive")
    field = torch.exp(-(x * x + y * y) / (2 * width**2)) * torch.cos(
        2 * math.pi * _rand(g, *frequency) * u + _rand(g, 0, 2 * math.pi)
    )
    return _colorize(_unit(field), channels, g)


@PRIORS.register("perlin")
def perlin(channels, size, g, resolutions=(2, 4, 8)):
    """2D gradient noise with a quintic fade; image size need not divide the grid."""
    r = int(resolutions[torch.randint(len(resolutions), (), generator=g).item()])
    if not 1 <= r <= size:
        raise ValueError("perlin resolution must lie in [1, image_size]")
    a = torch.rand(r + 1, r + 1, generator=g) * 2 * math.pi
    gradients = torch.stack((a.cos(), a.sin()), -1)
    v = torch.arange(size, dtype=torch.float32) * r / size
    y, x = torch.meshgrid(v, v, indexing="ij")
    ix, iy = x.long(), y.long()
    fx, fy = x - ix, y - iy

    def dot(dx, dy):
        grad = gradients[iy + dy, ix + dx]
        return grad[..., 0] * (fx - dx) + grad[..., 1] * (fy - dy)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    u, w = fade(fx), fade(fy)
    top = dot(0, 0) * (1 - u) + dot(1, 0) * u
    bottom = dot(0, 1) * (1 - u) + dot(1, 1) * u
    return _colorize(_unit(top * (1 - w) + bottom * w), channels, g)


class SyntheticDataset(Dataset):
    """A fixed virtual corpus: sample i is a pure function of (seed, i)."""

    def __init__(self, config):
        d, s = config["dataset"], config["synthetic_data"]
        self.channels, self.size = d["channels"], d["image_size"]
        self.seed = s.get("seed", config["experiment"]["seed"])
        self.n = s["num_samples"]
        if self.n < 1:
            raise ValueError("synthetic_data.num_samples must be positive")
        self.specs = s["priors"]
        self.names = [k for k, v in self.specs.items() if v.get("weight", 1) > 0]
        if not self.names or any(v.get("weight", 1) < 0 for v in self.specs.values()):
            raise ValueError("Prior weights must be nonnegative with positive enabled mass")
        self.weights = torch.tensor(
            [self.specs[k].get("weight", 1) for k in self.names], dtype=torch.float
        )
        for name in self.names:  # Fail at construction, not mid-epoch.
            self._draw(name, torch.Generator().manual_seed(0))

    def _draw(self, name, g):
        return PRIORS.resolve(name)(
            self.channels, self.size, g, **self.specs.get(name, {}).get("params", {})
        )

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if not 0 <= index < self.n:
            raise IndexError(index)
        g = torch.Generator().manual_seed((self.seed + index * 1000003) % (2**63 - 1))
        name = self.names[torch.multinomial(self.weights, 1, generator=g).item()]
        x = self._draw(name, g)
        if x.shape != (self.channels, self.size, self.size) or not torch.isfinite(x).all():
            raise ValueError(f"Invalid image generated by {name}")
        return {"image": x.clamp(0, 1), "prior": name}

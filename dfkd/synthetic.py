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
    a = _rand(g, 0, math.pi)
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


@PRIORS.register("pink")
def pink(channels, size, g):
    frequency = torch.fft.fftfreq(size)
    y, x = torch.meshgrid(frequency, frequency, indexing="ij")
    radius = (x.square() + y.square()).sqrt()
    radius[0, 0] = 1e-3
    field = torch.fft.ifft2(torch.fft.fft2(torch.randn(size, size, generator=g)) / radius).real
    return _colorize(_unit(field), channels, g)


@PRIORS.register("rect_patch")
def rect_patch(channels, size, g):
    canvas = linear_gradient(channels, size, g)
    width, height = torch.randint(max(1, size // 8), max(2, size // 2 + 1), (2,), generator=g).tolist()
    x = torch.randint(size - width + 1, (), generator=g).item()
    y = torch.randint(size - height + 1, (), generator=g).item()
    canvas[:, y:y + height, x:x + width] = torch.rand(channels, 1, 1, generator=g)
    return canvas


def _polygon_mask(vertices, size):
    axis = torch.linspace(0, 1, size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    inside = torch.zeros(size, size, dtype=torch.bool)
    previous = vertices[-1]
    for current in vertices:
        xi, yi = current
        xj, yj = previous
        inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        previous = current
    return inside


@PRIORS.register("closed_curve")
def closed_curve(channels, size, g):
    kind = torch.randint(3, (), generator=g).item()
    if kind == 0:
        n = torch.randint(5, 11, (), generator=g).item()
        angles = torch.sort(torch.rand(n, generator=g) * 2 * math.pi).values
        radius = 0.15 + 0.3 * torch.rand(n, generator=g)
    else:
        n = torch.randint(30, 61, (), generator=g).item()
        angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        if kind == 1:
            radius = torch.full((n,), 0.3)
            for _ in range(torch.randint(2, 6, (), generator=g).item()):
                k = torch.randint(1, 7, (), generator=g).item()
                radius += _rand(g, 0.02, 0.08) * torch.cos(k * angles + _rand(g, 0, 2 * math.pi))
        else:
            lobes = torch.randint(4, 9, (), generator=g).item()
            outer, inner = _rand(g, 0.3, 0.42), _rand(g, 0.1, 0.22)
            radius = inner + (outer - inner) * (0.5 + 0.5 * torch.cos(angles * lobes))
    vertices = torch.stack((0.5 + radius * angles.cos(), 0.5 + radius * angles.sin()), dim=1)
    return _colorize(_polygon_mask(vertices, size).float(), channels, g)


@PRIORS.register("random_circles")
def random_circles(channels, size, g, min_blobs=1, max_blobs=8):
    axis = torch.linspace(0, 1, size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    canvas = torch.rand(channels, 1, 1, generator=g).expand(channels, size, size).clone()
    for _ in range(torch.randint(min_blobs, max_blobs + 1, (), generator=g).item()):
        cx, cy, radius = _rand(g), _rand(g), _rand(g, 0.05, 0.35)
        mask = (x - cx).square() + (y - cy).square() <= radius ** 2
        canvas[:, mask] = torch.rand(channels, 1, generator=g)
    return canvas


class SyntheticDataset(Dataset):
    """A fixed virtual corpus: sample i is a pure function of (seed, i)."""

    def __init__(self, config):
        d, s = config["dataset"], config["synthetic_data"]
        self.channels, self.size = d["channels"], d["image_size"]
        self.seed = s.get("seed", config["experiment"]["seed"])
        self.n = s["num_samples"]
        self.group_size = s.get("group_size", 1)
        if not isinstance(self.group_size, int) or self.group_size < 1:
            raise ValueError("synthetic_data.group_size must be a positive integer")
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
        family_rng = torch.Generator().manual_seed(
            (self.seed + (index // self.group_size) * 1000003) % (2**63 - 1)
        )
        name = self.names[torch.multinomial(self.weights, 1, generator=family_rng).item()]
        x = self._draw(name, g)
        if x.shape != (self.channels, self.size, self.size) or not torch.isfinite(x).all():
            raise ValueError(f"Invalid image generated by {name}")
        return {"image": x.clamp(0, 1), "prior": name}


class AugmentedSyntheticDataset(Dataset):
    """Fixed noise bank with fresh per-access augmentation in DataLoader workers."""

    def __init__(self, config):
        self.base = SyntheticDataset(config)
        self.augmentation = config["augmentation"]
        self.dataset = config["dataset"]
        self._augment = None
        self.images = None
        if config["synthetic_data"].get("materialize", True):
            self.images = torch.empty(len(self.base), self.base.channels, self.base.size, self.base.size)
            for i in range(len(self.base)):
                self.images[i].copy_(self.base[i]["image"])

    def __len__(self):
        return len(self.base)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_augment"] = None  # Transform closures are recreated in spawn workers.
        return state

    def __getitem__(self, index):
        from dfkd.augment import Augment

        if self._augment is None:
            self._augment = Augment(self.augmentation, self.dataset)
        image = self.base[index]["image"] if self.images is None else self.images[index]
        return self._augment(image.clone())

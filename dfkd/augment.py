"""Query augmentation: the only source of diversity the student ever sees.

Protocol, applied to every synthetic image independently:

  stage 1 - geometric: RandomCrop(padding) then RandomHorizontalFlip, always applied
  stage 2 - RandAug:   n of the 17 photometric/geometric operations below, drawn
                       without replacement (17-choose-n) and applied sequentially

Horizontal flip is disabled for MNIST, where a mirrored digit is a different digit;
that dataset therefore runs one geometric transform rather than two.
"""

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from dfkd.registry import Registry

OPERATIONS = Registry()
OPERATIONS.update(
    rotation=T.RandomRotation,
    perspective=T.RandomPerspective,
    random_resized_crop=T.RandomResizedCrop,
    color_jitter=T.ColorJitter,
    grayscale=T.RandomGrayscale,
    autocontrast=T.RandomAutocontrast,
    inversion=T.RandomInvert,
)


@OPERATIONS.register("histogram_equalization")
def histogram_equalization(p=1.0):
    def apply(x):
        if torch.rand(()) >= p:
            return x
        return F.equalize((x.clamp(0, 1) * 255).to(torch.uint8)).float() / 255

    return apply


def _sample(value):
    if isinstance(value, (list, tuple)):
        return torch.empty(()).uniform_(*value).item()
    return value


@OPERATIONS.register("posterization")
def posterization(bits=(3, 4, 5, 6), p=1.0):
    def apply(x):
        if p < 1 and torch.rand(()) >= p:
            return x
        chosen = bits[torch.randint(len(bits), ()).item()] if isinstance(bits, (list, tuple)) else bits
        return F.posterize((x.clamp(0, 1) * 255).to(torch.uint8), chosen).float() / 255
    return apply


@OPERATIONS.register("gaussian_noise")
def gaussian_noise(std=(0.01, 0.08), mean=0.0):
    return lambda x: (x + torch.randn_like(x) * _sample(std) + mean).clamp(0, 1)


@OPERATIONS.register("salt_and_pepper_noise")
def salt_and_pepper_noise(probability=(0.01, 0.06)):
    def apply(x):
        prob = _sample(probability)
        mask = torch.rand(1, *x.shape[-2:])
        return torch.where(mask < prob / 2, 1.0, torch.where(mask > 1 - prob / 2, 0.0, x))
    return apply


@OPERATIONS.register("cutout")
def cutout(scale=(0.02, 0.25), ratio=(0.3, 3.3), fill=0.0):
    return T.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=fill)


@OPERATIONS.register("random_color_region_erasing")
def random_color_region_erasing(min_size=4, max_size=12):
    def apply(x):
        c, h, w = x.shape
        rh = torch.randint(min(min_size, h), min(max_size, h) + 1, ()).item()
        rw = torch.randint(min(min_size, w), min(max_size, w) + 1, ()).item()
        y = torch.randint(h - rh + 1, ()).item()
        z = torch.randint(w - rw + 1, ()).item()
        out = x.clone()
        out[:, y:y + rh, z:z + rw] = torch.rand(c, 1, 1)
        return out
    return apply


def gaussian_blur(kernel_sizes=(3, 5), sigma=(0.1, 2.2)):
    return lambda x: F.gaussian_blur(x, kernel_sizes[torch.randint(len(kernel_sizes), ()).item()], _sample(sigma))


def sharpness(sharpness_factor=(0.0, 3.5)):
    return lambda x: F.adjust_sharpness(x, _sample(sharpness_factor))


def solarization(threshold=(0.3, 0.9)):
    return lambda x: F.solarize(x, _sample(threshold))


def affine(degrees=12, translate_pixels=2, scale=(0.82, 1.18), shear=12):
    return lambda x: F.affine(
        x, _sample((-degrees, degrees)),
        torch.randint(-translate_pixels, translate_pixels + 1, (2,)).tolist(),
        _sample(scale), _sample((-shear, shear)),
    )


OPERATIONS.update(gaussian_blur=gaussian_blur, sharpness=sharpness,
                  solarization=solarization, affine=affine)


class Augment:
    """Two-stage query augmentation; every image draws its own operations."""

    def __init__(self, config, dataset):
        self.enabled = config["enabled"]
        self.k = config["num_random_ops"]
        self.without_replacement = config.get("without_replacement", True)
        size, channels = dataset["image_size"], dataset["channels"]

        geometric = config.get("geometric", {})
        crop = dict(geometric.get("crop", {}))
        if crop.get("padding_mode") == "reflect" and crop.get("padding", 0) >= size:
            raise ValueError("Reflect padding must be smaller than image_size")
        self.geometric = [T.RandomCrop(size, **crop)] if geometric.get("enabled", True) else []
        if geometric.get("enabled", True) and geometric.get("horizontal_flip", False):
            self.geometric.append(T.RandomHorizontalFlip(p=0.5))

        self.operations = []
        self.names = []
        for name, params in config["operations"].items():
            params = dict(params or {})
            if channels == 1 and name == "color_jitter":
                params.pop("saturation", None)
                params.pop("hue", None)
            if name == "random_resized_crop":
                params["size"] = size
            self.operations.append(OPERATIONS.resolve(name)(**params))
            self.names.append(name)
        if self.k < 0 or (self.without_replacement and self.k > len(self.operations)):
            raise ValueError(
                f"num_random_ops={self.k} cannot exceed the {len(self.operations)} "
                "registered operations when sampling without replacement"
            )
        if self.k and not self.operations:
            raise ValueError("num_random_ops > 0 requires at least one operation")

    def __call__(self, x):
        if not self.enabled:
            return x.clone()
        for op in self.geometric:  # Stage 1, in order.
            x = op(x)
        if self.k:  # Stage 2, n of N drawn per image, applied in the drawn order.
            indices = (
                torch.randperm(len(self.operations))[: self.k]
                if self.without_replacement
                else torch.randint(len(self.operations), (self.k,))
            )
            for i in indices.tolist():
                x = self.operations[i](x).clamp(0, 1)
        return x.clamp(0, 1)

    def batch(self, x):
        return torch.stack([self(image) for image in x])

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
    affine=T.RandomAffine,
    perspective=T.RandomPerspective,
    random_resized_crop=T.RandomResizedCrop,
    color_jitter=T.ColorJitter,
    grayscale=T.RandomGrayscale,
    gaussian_blur=T.GaussianBlur,
    sharpness=T.RandomAdjustSharpness,
    autocontrast=T.RandomAutocontrast,
    solarization=T.RandomSolarize,
    inversion=T.RandomInvert,
)


@OPERATIONS.register("histogram_equalization")
def histogram_equalization(p=1.0):
    def apply(x):
        if torch.rand(()) >= p:
            return x
        return F.equalize((x * 255).round().to(torch.uint8)).float() / 255

    return apply


@OPERATIONS.register("posterization")
def posterization(bits=4, p=1.0):
    def apply(x):
        if torch.rand(()) >= p:
            return x
        return F.posterize((x * 255).round().to(torch.uint8), bits).float() / 255

    return apply


@OPERATIONS.register("gaussian_noise")
def gaussian_noise(std=0.05, mean=0.0):
    if std < 0:
        raise ValueError("Noise std must be nonnegative")
    return lambda x: (x + torch.randn_like(x) * std + mean).clamp(0, 1)


@OPERATIONS.register("salt_and_pepper_noise")
def salt_and_pepper_noise(probability=0.03):
    if not 0 <= probability <= 1:
        raise ValueError("Noise probability must be in [0, 1]")

    def apply(x):
        mask = torch.rand_like(x)
        return torch.where(
            mask < probability / 2, 0.0, torch.where(mask > 1 - probability / 2, 1.0, x)
        )

    return apply


def _erase(fraction, fill):
    if not 0 < fraction <= 1:
        raise ValueError("Erasing fraction must be in (0, 1]")

    def apply(x):
        c, h, w = x.shape
        rh, rw = max(1, int(h * fraction)), max(1, int(w * fraction))
        y = torch.randint(h - rh + 1, ()).item()
        z = torch.randint(w - rw + 1, ()).item()
        x = x.clone()
        x[:, y : y + rh, z : z + rw] = torch.rand(c, 1, 1) if fill is None else fill
        return x

    return apply


@OPERATIONS.register("cutout")
def cutout(fraction=0.25, fill=0.0):
    return _erase(fraction, fill)


@OPERATIONS.register("random_color_region_erasing")
def random_color_region_erasing(fraction=0.25):
    return _erase(fraction, None)


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
        self.geometric = [T.RandomCrop(size, **crop)]
        if geometric.get("horizontal_flip", False):
            self.geometric.append(T.RandomHorizontalFlip(p=0.5))

        self.operations = []
        self.names = []
        for name, params in config["operations"].items():
            params = dict(params or {})
            if channels == 1 and name == "color_jitter":
                if params.get("saturation", 0) or params.get("hue", 0):
                    raise ValueError("Grayscale color_jitter: disable hue and saturation")
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
                x = self.operations[i](x)
        return x.clamp(0, 1)

    def batch(self, x):
        return torch.stack([self(image) for image in x])

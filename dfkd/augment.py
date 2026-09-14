"""Per-image CPU augmentation: the only source of diversity the student sees."""

import torch
from torchvision import transforms as T

from dfkd.registry import Registry

OPERATIONS = Registry()
OPERATIONS.update(
    rotation=T.RandomRotation,
    affine=T.RandomAffine,
    perspective=T.RandomPerspective,
    random_resized_crop=T.RandomResizedCrop,
    color_jitter=T.ColorJitter,
    gaussian_blur=T.GaussianBlur,
    sharpness=T.RandomAdjustSharpness,
    autocontrast=T.RandomAutocontrast,
    inversion=T.RandomInvert,
)


@OPERATIONS.register("gaussian_noise")
def gaussian_noise(std=0.05):
    if std < 0:
        raise ValueError("Noise std must be nonnegative")
    return lambda x: (x + torch.randn_like(x) * std).clamp(0, 1)


@OPERATIONS.register("cutout")
def cutout(fraction=0.25, fill=0.0):
    if not 0 < fraction <= 1:
        raise ValueError("Cutout fraction must be in (0, 1]")

    def apply(x):
        _, h, w = x.shape
        rh, rw = max(1, int(h * fraction)), max(1, int(w * fraction))
        y = torch.randint(h - rh + 1, ()).item()
        z = torch.randint(w - rw + 1, ()).item()
        x = x.clone()
        x[:, y : y + rh, z : z + rw] = fill
        return x

    return apply


class Augment:
    """Applies `num_random_ops` operations drawn independently for each image."""

    def __init__(self, config, dataset):
        self.enabled = config["enabled"]
        self.k = config["num_random_ops"]
        size, channels = dataset["image_size"], dataset["channels"]
        self.always = [T.RandomCrop(size, padding=config.get("crop_padding", 0))]
        if config.get("horizontal_flip", False):
            self.always.append(T.RandomHorizontalFlip())
        self.operations = []
        for name, params in config["operations"].items():
            params = dict(params or {})
            if channels == 1 and name == "color_jitter":
                params.pop("saturation", None)
                params.pop("hue", None)
            if name == "random_resized_crop":
                params["size"] = size
            self.operations.append(OPERATIONS.resolve(name)(**params))
        if self.k < 0 or self.k > len(self.operations):
            raise ValueError("num_random_ops must be between 0 and the number of operations")

    def __call__(self, x):
        if not self.enabled:
            return x.clone()
        for op in self.always:
            x = op(x)
        for i in torch.randperm(len(self.operations))[: self.k].tolist():
            x = self.operations[i](x)
        return x.clamp(0, 1)

    def batch(self, x):
        return torch.stack([self(image) for image in x])

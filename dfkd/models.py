"""Model zoo. Register a factory here to add an architecture; no branching elsewhere."""

import torch
from torch import nn

from dfkd.registry import Registry

MODELS = Registry()


class LeNet(nn.Module):
    def __init__(self, channels, num_classes, image_size, widths=(6, 16), hidden=(120, 84)):
        super().__init__()
        a, b = widths
        self.features = nn.Sequential(
            nn.Conv2d(channels, a, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(a, b, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 25, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS.register("lenet5")(LeNet)


@MODELS.register("lenet5_half")
def lenet5_half(**kwargs):
    return LeNet(**({"widths": (3, 8)} | kwargs))


def _resnet(depth, channels, num_classes, image_size):
    """torchvision ResNet with the small-image stem (3x3 conv, no max pool)."""
    from torchvision.models import resnet18, resnet34

    m = {18: resnet18, 34: resnet34}[depth](num_classes=num_classes)
    m.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


@MODELS.register("resnet18")
def resnet18(**kwargs):
    return _resnet(18, **kwargs)


@MODELS.register("resnet34")
def resnet34(**kwargs):
    return _resnet(34, **kwargs)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def build_model(spec, dataset):
    kwargs = {k: dataset[k] for k in ("channels", "num_classes", "image_size")}
    model = MODELS.resolve(spec["architecture"])(**kwargs, **spec.get("kwargs", {}))
    with torch.no_grad():
        model.eval()
        shape = (2, dataset["channels"], dataset["image_size"], dataset["image_size"])
        if model(torch.zeros(shape)).shape != (2, dataset["num_classes"]):
            raise ValueError("Model output does not match the dataset class count")
    return model


def freeze(model):
    return model.eval().requires_grad_(False)

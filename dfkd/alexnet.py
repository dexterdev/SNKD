"""CIFAR-10 AlexNet pair from Nayak et al., ICML 2019, supplement Table 2.

https://proceedings.mlr.press/v97/nayak19a/nayak19a-supp.pdf
BatchNorm hyperparameters are unspecified in the table; use PyTorch defaults
(affine=True, eps=1e-5, momentum=0.1). Return logits for the shared CE/KL losses.
"""

from torch import nn


class AlexNetCIFAR(nn.Module):
    def __init__(self, channels, num_classes, image_size, half=False):
        super().__init__()
        if (channels, image_size, num_classes) != (3, 32, 10):
            raise ValueError("This AlexNet pair implements the 32x32 RGB CIFAR-10 architecture")
        widths = (24, 64, 96, 96, 64) if half else (48, 128, 192, 192, 128)
        hidden = (256, 128) if half else (512, 256)
        layers = []
        incoming = channels
        for index, outgoing in enumerate(widths):
            kernel = 5 if index < 2 else 3
            conv = nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2)
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 1.0 if index in (1, 3, 4) else 0.0)
            layers.extend([conv, nn.ReLU()])
            if index < 2:
                # TF depth_radius=2 sums 5 channels; PyTorch divides alpha by size.
                layers.append(nn.LocalResponseNorm(5, alpha=5e-4, beta=0.75, k=1.0))
            if index in (0, 1, 4):
                layers.append(nn.MaxPool2d(3, stride=2))
            layers.append(nn.BatchNorm2d(outgoing))
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1] * 3 * 3, hidden[0]), nn.ReLU(),
            nn.Dropout(0.5), nn.BatchNorm1d(hidden[0]),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Dropout(0.5), nn.BatchNorm1d(hidden[1]),
            nn.Linear(hidden[1], num_classes),
        )
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

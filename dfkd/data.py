"""Real data access. Real training images are reachable only for teacher training."""

from torch.utils.data import DataLoader

from dfkd.registry import Registry

DATASETS = Registry(
    mnist=dict(constructor="MNIST", channels=1, num_classes=10),
    fashionmnist=dict(constructor="FashionMNIST", channels=1, num_classes=10),
    cifar10=dict(constructor="CIFAR10", channels=3, num_classes=10),
    cifar100=dict(constructor="CIFAR100", channels=3, num_classes=100),
)


def real_dataset(config, split, *, purpose):
    if split == "train" and purpose != "teacher_training":
        raise PermissionError("Real training data is forbidden during student distillation")
    if split == "test" and purpose != "evaluation":
        raise PermissionError("Real test data is only available for evaluation")
    from torchvision import datasets, transforms

    d = config["dataset"]
    constructor = getattr(datasets, DATASETS.resolve(d["name"])["constructor"])
    transform = transforms.Compose(
        [transforms.Resize((d["image_size"], d["image_size"])), transforms.ToTensor()]
    )
    return constructor(
        d["data_root"], train=split == "train", download=d.get("download", True), transform=transform
    )


def test_loader(config):
    return DataLoader(
        real_dataset(config, "test", purpose="evaluation"),
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
    )


def normalize(x, dataset):
    n = dataset["normalization"]
    mean = x.new_tensor(n["mean"])[None, :, None, None]
    std = x.new_tensor(n["std"])[None, :, None, None]
    return (x - mean) / std

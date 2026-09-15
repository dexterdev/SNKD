"""CIFAR AlexNet Table 2 dimensions and shared training integration."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dfkd.config import load, validate
from dfkd.models import build_model, parameter_count

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("role,widths,hidden,total,base", [
    ("teacher", [48, 128, 192, 192, 128], [512, 256, 10], 1659178, 1656266),
    ("student", [24, 64, 96, 96, 64], [256, 128, 10], 417434, 415978),
])
def test_table2_architecture(role, widths, hidden, total, base):
    torch.set_num_threads(1)
    c = validate(load(ROOT / "configs/cifar10_alexnet.yaml"))
    model = build_model(c[role], c["dataset"])
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    fcs = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert [m.out_channels for m in convs] == widths
    assert [m.kernel_size for m in convs] == [(5, 5)] * 2 + [(3, 3)] * 3
    assert [m.out_features for m in fcs] == hidden
    assert parameter_count(model) == total
    assert sum(p.numel() for m in convs + fcs for p in m.parameters()) == base
    assert sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()) == 7
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [0.5, 0.5]
    sizes = []
    handles = [m.register_forward_hook(lambda m, args, out: sizes.append(out.shape[-1]))
               for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    for handle in handles:
        handle.remove()
    assert sizes == [15, 7, 3]
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for conv, bias in zip(convs, [0, 1, 0, 1, 1]):
        assert torch.all(conv.bias == bias)


def test_lrn_matches_tensorflow_depth_radius_two():
    c = load(ROOT / "configs/cifar10_alexnet.yaml")
    model = build_model(c["teacher"], c["dataset"])
    norms = [m for m in model.modules() if isinstance(m, nn.LocalResponseNorm)]
    assert len(norms) == 2
    x = torch.randn(2, 9, 3, 3) * 10
    squared_sum = torch.stack([
        x[:, max(0, i - 2):i + 3].square().sum(dim=1) for i in range(9)
    ], dim=1)
    expected = x / (1 + 1e-4 * squared_sum).pow(0.75)
    for norm in norms:
        torch.testing.assert_close(norm(x), expected)


def test_alexnet_teacher_and_student_pipeline(tmp_path, monkeypatch):
    import dfkd.teacher as teacher_module
    from dfkd.train import train_student

    torch.set_num_threads(1)
    c = load(ROOT / "configs/cifar10_alexnet.yaml")
    c["experiment"]["output_dir"] = str(tmp_path / "runs")
    c["teacher"]["checkpoint"] = str(tmp_path / "teacher.pt")
    c["teacher_training"].update(epochs=1, batch_size=4, augmentation={"enabled": False})
    c["training"].update(epochs=1, batch_size_schedule=[4], device="cpu")
    c["training"]["runtime"]["num_workers"] = 0
    c["synthetic_data"]["num_samples"] = 8
    c["augmentation"]["enabled"] = False
    c["evaluation"]["batch_size"] = 4
    data = TensorDataset(torch.rand(20, 3, 32, 32), torch.arange(20) % 10)
    data.targets = torch.arange(20) % 10
    testing = DataLoader(data, batch_size=4)
    monkeypatch.setattr(teacher_module, "real_dataset", lambda *a, **kw: data)
    monkeypatch.setattr(teacher_module, "test_loader", lambda *a: testing)
    checkpoint = teacher_module.train_teacher(c)
    assert checkpoint.is_file()
    run = train_student(c, evaluation_loader=testing)
    restored = build_model(c["student"], c["dataset"])
    restored.load_state_dict(torch.load(run / "best.pt", weights_only=True), strict=True)
    assert (run / "metrics.csv").is_file()
    assert (run / "results.json").is_file()

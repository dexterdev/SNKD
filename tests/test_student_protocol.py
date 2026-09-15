"""Notebook protocol regression tests, including phase edges and fresh queries."""

import copy
import csv
import json
import pickle
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfkd.config import load, validate
from dfkd.distill import kl
from dfkd.runtime import StudentRuntime
from dfkd.schedule import Schedule
from dfkd.synthetic import AugmentedSyntheticDataset, SyntheticDataset

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["mnist", "fashionmnist", "cifar10", "cifar100", "cifar10_vit", "cifar100_vit"]


@pytest.mark.parametrize("name", CONFIGS)
def test_all_configs_inherit_notebook_protocol(name):
    c = validate(load(ROOT / f"configs/{name}.yaml"))
    assert c["training"]["batch_size_schedule"] == [16, 64, 256, 1024]
    assert c["training"]["phase_epochs"] == 50
    assert c["synthetic_data"]["num_samples"] == 150000
    assert c["synthetic_data"]["group_size"] == 100
    assert set(c["synthetic_data"]["priors"]) == {
        "linear_gradient", "perlin", "uniform", "gabor", "checkerboard", "pink",
        "rect_patch", "closed_curve", "random_circles",
    }
    assert c["augmentation"]["num_random_ops"] == 8
    assert c["distillation"] == {"loss": "kl_divergence", "temperature": 20.0}


def test_notebook_lr_and_batch_boundaries():
    c = load(ROOT / "configs/cifar100.yaml")
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], **{
        k: v for k, v in c["optimizer"].items() if k != "name"
    })
    scheduler = Schedule(opt, c["scheduler"], c["training"])
    for epoch, batch, lr in [(0, 16, .01), (49, 16, 1e-7), (50, 64, .01),
                              (99, 64, 1e-7), (100, 256, .01), (149, 256, 1e-7),
                              (150, 1024, .01), (199, 1024, 1e-7)]:
        assert scheduler.step(epoch)[1] == batch
        assert opt.param_groups[0]["lr"] == pytest.approx(lr)
    # Shortening a run truncates phases rather than compressing the schedule.
    c["training"]["epochs"] = 75
    scheduler = Schedule(opt, c["scheduler"], c["training"])
    assert scheduler.step(49)[1] == 16
    assert scheduler.step(74)[1] == 64


def test_noise_bank_is_fixed_grouped_and_freshly_augmented():
    c = load(ROOT / "configs/cifar100.yaml")
    c["synthetic_data"].update(num_samples=12, group_size=4)
    base = SyntheticDataset(c)
    for start in (0, 4, 8):
        assert len({base[i]["prior"] for i in range(start, start + 4)}) == 1
    bank = AugmentedSyntheticDataset(c)
    original = bank.images.clone()
    views = [bank[0] for _ in range(3)]
    assert torch.equal(original, bank.images)
    assert not torch.equal(views[0], views[1])
    assert bank.images.shape == (12, 3, 32, 32)
    # Spawn-safe even after lazy transforms have been initialized.
    restored = pickle.loads(pickle.dumps(bank))
    assert restored[0].shape == (3, 32, 32)
    c["synthetic_data"]["materialize"] = False
    c["augmentation"]["enabled"] = False
    virtual = AugmentedSyntheticDataset(c)
    assert torch.equal(virtual[0], original[0])


def test_kl_matches_notebook_and_student_only_backprop():
    student = torch.randn(3, 10, requires_grad=True)
    teacher = torch.randn(3, 10)
    expected = torch.nn.functional.kl_div(
        (student / 20).log_softmax(-1), (teacher / 20).softmax(-1), reduction="batchmean"
    ) * 20**2
    actual = kl(student, teacher, 20)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(student.grad).all()


def test_student_queries_same_augmented_view_and_drops_tail(tmp_path, monkeypatch):
    import dfkd.train as module

    c = load(ROOT / "configs/mnist.yaml")
    c["experiment"]["output_dir"] = str(tmp_path)
    c["synthetic_data"].update(num_samples=10, group_size=1)
    c["training"].update(epochs=2, phase_epochs=1, batch_size_schedule=[4, 8], device="cpu")
    c["training"]["runtime"]["num_workers"] = 0
    c["augmentation"]["num_random_ops"] = 8
    teacher_views, student_views = [], []

    class Probe(torch.nn.Module):
        def __init__(self, views):
            super().__init__()
            self.linear = torch.nn.Linear(28 * 28, 10)
            self.views = views

        def forward(self, x):
            self.views.append(x.detach().clone())
            return self.linear(x.flatten(1))

    teacher = Probe(teacher_views).eval().requires_grad_(False)
    student = Probe(student_views)
    before = copy.deepcopy(teacher.state_dict())
    monkeypatch.setattr(module, "load_teacher", lambda *a: teacher)
    monkeypatch.setattr(module, "build_model", lambda *a: student)
    monkeypatch.setattr(module, "evaluate", lambda *a: {"accuracy": 50., "loss": 1.})
    run = module.train_student(c, evaluation_loader=[])
    assert len(teacher_views) == len(student_views) == 3
    for x, y in zip(teacher_views, student_views):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in before.items())
    result = json.loads((run / "results.json").read_text())
    assert result["teacher_queries"] == 16  # 2*4 then 1*8, tails discarded.
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["batch_size"]) for r in rows] == [4, 8]
    assert [int(r["train_samples"]) for r in rows] == [8, 8]


def test_worker_loader_and_cpu_precision():
    c = load(ROOT / "configs/mnist.yaml")
    c["synthetic_data"]["num_samples"] = 4
    bank = AugmentedSyntheticDataset(c)
    runtime = StudentRuntime(torch.device("cpu"), {"num_workers": 1})
    loader = runtime.loader(bank, 2, True, 42)
    loader.timeout = 15
    for _ in range(2):
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].dtype == torch.float32
        assert torch.isfinite(batches[0]).all()
    del loader
    assert runtime.dtype is None and not runtime.scaler.is_enabled()

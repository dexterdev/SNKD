import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfkd.augment import OPERATIONS, Augment
from dfkd.config import load, validate
from dfkd.data import real_dataset
from dfkd.models import build_model, parameter_count
from dfkd.schedule import Schedule, phase_at
from dfkd.synthetic import PRIORS, SyntheticDataset
from dfkd.train import train_student

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path):
    c = load(ROOT / "configs/mnist.yaml")
    c["experiment"]["output_dir"] = str(tmp_path / "runs")
    c["teacher"]["checkpoint"] = str(tmp_path / "teacher.pt")
    c["synthetic_data"]["num_samples"] = 8
    c["training"].update(epochs=2, batch_size_schedule=[4], device="cpu")
    c["evaluation"]["batch_size"] = 4
    c["augmentation"]["num_random_ops"] = 2
    return validate(c)


@pytest.fixture
def teacher_checkpoint(config):
    torch.manual_seed(7)
    model = build_model(config["teacher"], config["dataset"])
    torch.save(model.state_dict(), config["teacher"]["checkpoint"])
    return model


@pytest.fixture
def heldout():
    # Artificial evaluation fixture so tests never download or touch real data.
    g = torch.Generator().manual_seed(99)
    data = TensorDataset(torch.rand(8, 1, 28, 28, generator=g), torch.arange(8) % 10)
    return DataLoader(data, batch_size=4)


@pytest.mark.parametrize("channels,size", [(1, 28), (3, 32)])
@pytest.mark.parametrize("name", sorted(PRIORS))
def test_priors_are_valid_and_deterministic(config, channels, size, name):
    config["dataset"].update(channels=channels, image_size=size)
    config["synthetic_data"]["priors"] = {name: {"weight": 1.0}}
    sample = SyntheticDataset(config)[2]
    assert sample["prior"] == name
    assert sample["image"].shape == (channels, size, size)
    assert torch.isfinite(sample["image"]).all()
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 1
    assert torch.equal(sample["image"], SyntheticDataset(config)[2]["image"])


def test_prior_weights_are_respected(config):
    config["synthetic_data"].update(
        num_samples=2000, priors={"uniform": {"weight": 1}, "gaussian": {"weight": 3}}
    )
    data = SyntheticDataset(config)
    gaussian = sum(data[i]["prior"] == "gaussian" for i in range(len(data))) / len(data)
    assert 0.70 < gaussian < 0.80


def test_sampling_does_not_touch_global_rng(config):
    state = torch.get_rng_state().clone()
    SyntheticDataset(config)[0]
    assert torch.equal(state, torch.get_rng_state())


def test_augmentation_stays_in_range_and_changes_the_image(config):
    augment = Augment(config["augmentation"], config["dataset"])
    x = SyntheticDataset(config)[0]["image"]
    y = augment(x)
    assert y.shape == x.shape
    assert 0 <= y.min() <= y.max() <= 1
    assert not torch.equal(x, y)


def test_config_validation_rejects_bad_values(config):
    config["distillation"]["temperature"] = 0
    with pytest.raises(ValueError, match="temperature"):
        validate(config)


def test_real_training_data_is_blocked_for_students(config):
    with pytest.raises(PermissionError):
        real_dataset(config, "train", purpose="distillation")


def test_student_is_smaller_than_teacher(config):
    teacher = build_model(config["teacher"], config["dataset"])
    student = build_model(config["student"], config["dataset"])
    assert parameter_count(teacher) == 61706
    assert parameter_count(student) < parameter_count(teacher)


def test_distillation_end_to_end(config, teacher_checkpoint, heldout, monkeypatch):
    import dfkd.train

    def forbidden(*args, **kwargs):
        raise AssertionError("Student distillation must not construct a real dataset")

    monkeypatch.setattr(dfkd.train, "real_dataset", forbidden)
    run = train_student(config, evaluation_loader=heldout)
    for name in ("config.yaml", "metrics.csv", "results.json", "last.pt"):
        assert (run / name).is_file()
    results = json.loads((run / "results.json").read_text())
    assert results["teacher_parameters"] == 61706
    assert results["teacher_queries"] == 16  # 8 samples x 2 epochs
    assert results["final_student_accuracy"] is not None


def test_randaug_pool_has_seventeen_operations():
    assert len(OPERATIONS) == 17


@pytest.mark.parametrize("k", [0, 1, 3, 17])
def test_augmentation_applies_n_of_seventeen_without_replacement(config, k, monkeypatch):
    config["augmentation"]["num_random_ops"] = k
    augment = Augment(config["augmentation"], config["dataset"])
    assert len(augment.operations) == 17
    drawn = []
    for index, name in enumerate(augment.names):
        augment.operations[index] = lambda x, name=name: (drawn.append(name), x)[1]
    augment(SyntheticDataset(config)[0]["image"])
    assert len(drawn) == k
    assert len(set(drawn)) == k  # 17-choose-k: no operation is drawn twice.


def test_num_random_ops_cannot_exceed_the_pool(config):
    config["augmentation"]["num_random_ops"] = 18
    with pytest.raises(ValueError, match="cannot exceed"):
        Augment(config["augmentation"], config["dataset"])


def test_geometric_stage_runs_before_the_random_ops(config):
    order = []
    augment = Augment(config["augmentation"], config["dataset"])
    augment.geometric = [lambda x: (order.append("geometric"), x)[1]]
    for index in range(len(augment.operations)):
        augment.operations[index] = lambda x: (order.append("randaug"), x)[1]
    augment(SyntheticDataset(config)[0]["image"])
    assert order[0] == "geometric"
    assert order.count("geometric") == 1
    assert order[1:] == ["randaug"] * config["augmentation"]["num_random_ops"]


@pytest.mark.parametrize(
    "name,flip", [("mnist", False), ("fashionmnist", True), ("cifar10", True)]
)
def test_flip_is_configured_per_dataset(name, flip):
    c = load(ROOT / f"configs/{name}.yaml")
    assert c["augmentation"]["geometric"]["horizontal_flip"] is flip
    augment = Augment(c["augmentation"], c["dataset"])
    # Stage 1 is crop plus, where enabled, the flip: two geometric transforms.
    assert len(augment.geometric) == (2 if flip else 1)


def test_fashionmnist_config_uses_the_lenet_pair():
    c = validate(load(ROOT / "configs/fashionmnist.yaml"))
    assert c["dataset"]["name"] == "fashionmnist"
    assert (c["teacher"]["architecture"], c["student"]["architecture"]) == (
        "lenet5",
        "lenet5_half",
    )


def test_batch_size_phases_partition_the_epochs():
    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    assert phase_at(0, 200, sizes) == (0, 0, 25, 16)
    assert phase_at(24, 200, sizes) == (0, 0, 25, 16)
    assert phase_at(25, 200, sizes) == (1, 25, 25, 32)
    assert phase_at(199, 200, sizes) == (7, 175, 25, 2048)
    with pytest.raises(ValueError, match="outside"):
        phase_at(200, 200, sizes)
    # Earlier phases absorb the remainder when epochs do not divide evenly.
    assert [phase_at(e, 5, [8, 16])[3] for e in range(5)] == [8, 8, 8, 16, 16]


def test_lr_restarts_inside_each_batch_size_phase():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training = dict(epochs=4, batch_size_schedule=[2, 4])
    schedule = Schedule(optimizer, {"name": "cosine", "phase_resets": True}, training)
    seen = [(schedule.step(e), optimizer.param_groups[0]["lr"]) for e in range(4)]
    assert [batch for (_, batch), _ in seen] == [2, 2, 4, 4]
    assert [lr for _, lr in seen] == [0.1, 0.0, 0.1, 0.0]  # Cosine restarts at the boundary.


def test_lr_decays_once_across_training_without_phase_resets():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training = dict(epochs=4, batch_size_schedule=[2, 4])
    schedule = Schedule(optimizer, {"name": "cosine", "phase_resets": False}, training)
    rates = [(schedule.step(e), optimizer.param_groups[0]["lr"])[1] for e in range(4)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] == 0.1 and rates[-1] == 0.0


def test_singleton_final_minibatch_is_rejected(config):
    config["synthetic_data"]["num_samples"] = 9
    config["training"]["batch_size_schedule"] = [4]
    with pytest.raises(ValueError, match="single-sample"):
        validate(config)


def test_distillation_steps_the_batch_size(config, teacher_checkpoint, heldout):
    config["training"].update(epochs=2, batch_size_schedule=[4, 8])
    run = train_student(config, evaluation_loader=heldout)
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert [int(r["batch_size"]) for r in rows] == [4, 8]
    assert [int(r["phase"]) for r in rows] == [0, 1]
    results = json.loads((run / "results.json").read_text())
    assert results["batch_size_schedule"] == [4, 8]

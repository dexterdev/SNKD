import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfkd.augment import Augment
from dfkd.config import load, validate
from dfkd.data import real_dataset
from dfkd.models import build_model, parameter_count
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
    c["training"].update(epochs=2, batch_size=4, device="cpu")
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

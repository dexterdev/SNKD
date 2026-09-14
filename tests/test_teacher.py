"""Teacher protocol regression tests; all data is artificial and CPU-only."""

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfkd.config import load
from dfkd.metrics import ClassificationMetrics
from dfkd.teacher import EarlyStopping, stratified_split, train_teacher

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_uniform_binary_predictions():
    metrics = ClassificationMetrics(2, "cpu")
    metrics.update(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))
    result = metrics.compute()
    assert result["loss"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert result["accuracy"] == 50
    assert result["balanced_accuracy"] == 50
    assert result["top5_accuracy"] == 100
    assert result["ece"] == 0
    assert result["brier_score"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


def test_metrics_streaming_matches_full_batch():
    g = torch.Generator().manual_seed(12)
    logits = torch.randn(11, 3, generator=g)
    labels = torch.arange(11) % 3
    full, chunks = ClassificationMetrics(3, "cpu"), ClassificationMetrics(3, "cpu")
    full.update(logits, labels)
    for x, y in zip(logits.split(4), labels.split(4)):
        chunks.update(x, y)
    a, b = full.compute(), chunks.compute()
    for key in ("loss", "accuracy", "ece", "brier_score", "macro_f1"):
        assert a[key] == pytest.approx(b[key], abs=1e-7)
    assert a["confusion_matrix"] == b["confusion_matrix"]


def test_stratified_split_is_disjoint_complete_and_seeded():
    labels = torch.arange(100) % 10
    train, validation = stratified_split(labels, 0.2, 7)
    assert not set(train) & set(validation)
    assert sorted(train + validation) == list(range(100))
    assert torch.bincount(labels[validation]).tolist() == [2] * 10
    assert (train, validation) == stratified_split(labels, 0.2, 7)
    assert (train, validation) != stratified_split(labels, 0.2, 8)


def test_early_stopping_burn_in_delta_and_disabled():
    stop = EarlyStopping(dict(patience=2, min_epochs=3, min_delta=0.01))
    assert not stop.step(1.0, 1)
    assert not stop.step(0.995, 2)
    assert not stop.step(0.994, 3)
    assert stop.step(0.993, 4)
    disabled = EarlyStopping(dict(patience=0, min_epochs=0))
    assert all(not disabled.step(1.0, epoch) for epoch in range(1, 10))
    with pytest.raises(FloatingPointError):
        disabled.step(float("nan"), 10)


def test_teacher_selection_uses_validation_despite_test_reporting(tmp_path, monkeypatch, capsys):
    import dfkd.teacher as module

    torch.set_num_threads(1)
    c = load(ROOT / "configs/mnist.yaml")
    c["experiment"]["output_dir"] = str(tmp_path / "runs")
    c["teacher"]["checkpoint"] = str(tmp_path / "teacher.pt")
    c["training"]["device"] = "cpu"
    c["teacher_training"].update(
        epochs=4, batch_size=8, validation_fraction=0.25,
        augmentation={"enabled": False},
        early_stopping={"patience": 1, "min_epochs": 1, "min_delta": 0.0},
    )
    data = TensorDataset(torch.rand(40, 1, 28, 28), torch.arange(40) % 10)
    data.targets = torch.arange(40) % 10
    monkeypatch.setattr(module, "real_dataset", lambda *a, **kw: data)
    monkeypatch.setattr(module, "build_model",
                        lambda *a: torch.nn.Sequential(torch.nn.Flatten(),
                                                       torch.nn.Linear(784, 10)))
    seen, states = [], []
    actual_evaluate = module.evaluate
    losses = iter([0.3, 0.4])
    test_data = DataLoader(data, batch_size=8)
    test_losses = iter([0.9, 0.1, 0.9])

    def testing(config):
        assert seen == []
        seen.append("test_loaded")
        return test_data

    def evaluate(model, loader, *args, **kwargs):
        result = actual_evaluate(model, loader, *args, **kwargs)
        if loader is test_data:
            seen.append("test_evaluated")
            # Test prefers later epochs; validation must still choose epoch 1.
            result["loss"] = next(test_losses)
        else:
            seen.append("validation")
            result["loss"] = next(losses)
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return result

    monkeypatch.setattr(module, "test_loader", testing)
    monkeypatch.setattr(module, "evaluate", evaluate)
    output = train_teacher(c)
    assert seen == ["test_loaded"] + ["validation", "test_evaluated"] * 2 + ["test_evaluated"]
    console = capsys.readouterr().out
    epoch_lines = [line for line in console.splitlines() if line.startswith("Epoch ")]
    assert len(epoch_lines) == 2
    assert all("train_loss=" in line and "val_loss=" in line and "test_loss=" in line
               and "test_acc=" in line for line in epoch_lines)
    saved = torch.load(output, weights_only=True)
    assert all(torch.equal(saved[key], states[0][key]) for key in saved)
    run = next((tmp_path / "runs").iterdir())
    result = json.loads((run / "results.json").read_text())
    assert result["best_epoch"] == 1
    assert result["best_val_loss"] == 0.3
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "val_ece" in rows[0] and "train_samples_per_second" in rows[0]
    assert "test_loss" in rows[0] and "test_accuracy" in rows[0]
    assert rows[-1]["early_stopped"] == "True"
    details = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(details[0]["val_confusion_matrix"]) == 10
    assert (run / "best.pt").is_file()
    assert not any(path.is_dir() for path in run.iterdir())


def test_teacher_epochs_cli_override():
    from argparse import Namespace
    from dfkd.cli import resolve

    args = Namespace(config=str(ROOT / "configs/mnist.yaml"), command="train_teacher",
                     temperature=None, seed=None, num_samples=None, epochs=3, set=[])
    config = resolve(args)
    assert config["teacher_training"]["epochs"] == 3
    assert config["training"]["epochs"] == 200

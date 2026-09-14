"""Teacher training (real data) and student distillation (synthetic data only)."""

import csv
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dfkd.augment import Augment
from dfkd.config import validate
from dfkd.data import normalize, real_dataset, test_loader
from dfkd.distill import kl, objective
from dfkd.models import build_model, freeze, parameter_count
from dfkd.metrics import ClassificationMetrics
from dfkd.schedule import Schedule
from dfkd.synthetic import SyntheticDataset


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(config):
    name = config["training"]["device"]
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate(model, loader, dataset, device, teacher=None, temperature=1.0):
    model.eval()
    metrics = ClassificationMetrics(dataset["num_classes"], device)
    loss = 0.0
    for images, labels in loader:
        x = normalize(images.to(device), dataset)
        labels = labels.to(device)
        logits = model(x)
        metrics.update(logits, labels)
        if teacher is not None:
            loss += kl(logits, teacher(x), temperature).item() * len(labels)
    result = metrics.compute()
    result["distillation_loss"] = loss / result["samples"] if teacher is not None else None
    return result


def make_optimizer(model, config):
    options = dict(config)
    name = options.pop("name")
    return {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[name](
        model.parameters(), **options
    )


def make_run(config):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run = Path(config["experiment"]["output_dir"]) / f"{config['experiment']['name']}_{stamp}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return run


def train_teacher(config):
    """Train/tune the teacher on a held-out portion of the training split."""
    from dfkd.teacher import train_teacher as run_teacher

    return run_teacher(config)


def load_teacher(config, device):
    path = config["teacher"]["checkpoint"]
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Teacher checkpoint missing: {path}. Train a teacher first.")
    model = build_model(config["teacher"], config["dataset"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return freeze(model.to(device))


def train_student(config, *, evaluation_loader=None):
    """Distillation. Real data enters only through the held-out evaluation loader."""
    c = validate(copy.deepcopy(config))
    d, t = c["dataset"], c["training"]
    seed_all(c["experiment"]["seed"])
    device = get_device(c)
    teacher = load_teacher(c, device)
    student = build_model(c["student"], d).to(device)
    optimizer = make_optimizer(student, c["optimizer"])
    scheduler = Schedule(optimizer, c["scheduler"], t)
    data = SyntheticDataset(c)
    augment = Augment(c["augmentation"], d)
    evaluator = evaluation_loader if evaluation_loader is not None else test_loader(c)
    run = make_run(c)
    teacher_accuracy = evaluate(teacher, evaluator, d, device)["accuracy"]
    print(
        f"teacher {c['teacher']['architecture']} {parameter_count(teacher):,} params | "
        f"student {c['student']['architecture']} {parameter_count(student):,} params | "
        f"compression {parameter_count(teacher) / parameter_count(student):.2f}x"
    )
    history = []
    queries = 0
    best = -1.0
    for epoch in range(t["epochs"]):
        phase, batch_size = scheduler.step(epoch)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        student.train()
        total = seen = 0
        for batch in loader:
            x = normalize(augment.batch(batch["image"]).to(device), d)
            with torch.no_grad():
                logits = teacher(x)
            queries += len(x)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(student(x), logits, c["distillation"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect the learning rate/temperature")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            seen += len(x)
        measured = (epoch + 1) % c["evaluation"]["every"] == 0 or epoch + 1 == t["epochs"]
        accuracy = (
            evaluate(student, evaluator, d, device)["accuracy"] if measured else None
        )
        if accuracy is not None and accuracy > best:
            best = accuracy
            torch.save(student.state_dict(), run / "best.pt")
        history.append(
            dict(
                epoch=epoch + 1,
                phase=phase,
                batch_size=batch_size,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=total / seen,
                student_accuracy=accuracy,
            )
        )
        print(json.dumps(history[-1]))
    torch.save(student.state_dict(), run / "last.pt")
    with (run / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final = next(
        (r["student_accuracy"] for r in reversed(history) if r["student_accuracy"] is not None),
        None,
    )
    retention = (
        100 * final / teacher_accuracy if final is not None and teacher_accuracy else None
    )
    results = dict(
        experiment=c["experiment"]["name"],
        dataset=d["name"],
        teacher_architecture=c["teacher"]["architecture"],
        student_architecture=c["student"]["architecture"],
        teacher_parameters=parameter_count(teacher),
        student_parameters=parameter_count(student),
        teacher_accuracy=teacher_accuracy,
        best_student_accuracy=best if best >= 0 else None,
        final_student_accuracy=final,
        accuracy_retention_percent=retention,
        teacher_queries=queries,
        final_train_loss=history[-1]["train_loss"],
        batch_size_schedule=t["batch_size_schedule"],
        lr_schedule=c["scheduler"],
    )
    (run / "results.json").write_text(json.dumps(results, indent=2))
    return run

"""Single-run teacher training with validation-based checkpoint selection."""

import copy
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dfkd.config import validate
from dfkd.data import normalize, real_dataset, test_loader
from dfkd.metrics import ClassificationMetrics
from dfkd.models import build_model, parameter_count
from dfkd.schedule import Schedule
from dfkd.train import evaluate, get_device, make_optimizer, make_run, seed_all


def stratified_split(targets, fraction, seed):
    """Keep every class in both splits; indices refer only to training data."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    targets = torch.as_tensor(targets)
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in targets.unique(sorted=True):
        indices = (targets == label).nonzero(as_tuple=True)[0]
        if len(indices) < 2:
            raise ValueError("Each class needs at least two training samples")
        indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
        size = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation.extend(indices[:size])
        train.extend(indices[size:])
    if not train or not validation:
        raise ValueError("Training and validation splits must be nonempty")
    return train, validation


class EarlyStopping:
    """Significant NLL improvement resets patience; raw best is saved separately."""
    def __init__(self, config):
        self.patience = config.get("patience", 20)
        self.min_epochs = config.get("min_epochs", 60)
        self.min_delta = config.get("min_delta", 0.0001)
        self.reference = math.inf
        self.bad_epochs = 0
        if (not isinstance(self.patience, int) or self.patience < 0
                or not isinstance(self.min_epochs, int) or self.min_epochs < 0
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError("Invalid early-stopping patience, min_epochs or min_delta")

    def step(self, loss, epoch):
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        if loss < self.reference - self.min_delta:
            self.reference, self.bad_epochs = loss, 0
        elif epoch >= self.min_epochs:
            self.bad_epochs += 1
        return self.patience > 0 and epoch >= self.min_epochs and self.bad_epochs >= self.patience


def _validate_training(prep):
    for key in ("epochs", "batch_size"):
        if not isinstance(prep[key], int) or prep[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"teacher_training.{key} is invalid")
    if not 0 <= prep.get("label_smoothing", 0.0) < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    opt = prep["optimizer"]
    if not math.isfinite(opt["lr"]) or opt["lr"] <= 0:
        raise ValueError("Teacher learning rate must be finite and positive")
    if not math.isfinite(opt.get("weight_decay", 0)) or opt.get("weight_decay", 0) < 0:
        raise ValueError("Teacher weight decay must be finite and nonnegative")
    EarlyStopping(prep.get("early_stopping", {}))



class TrainingView(Dataset):
    """Apply optional mild crop/flip only to the training subset."""
    def __init__(self, data, config):
        from torchvision import transforms

        self.data = data
        options = config["teacher_training"].get("augmentation", {})
        ops = []
        if options.get("enabled", True):
            padding = options.get("crop_padding", 4 if config["dataset"]["channels"] == 3 else 0)
            if padding:
                ops.append(transforms.RandomCrop(config["dataset"]["image_size"],
                                                 padding=padding))
            flip = options.get("horizontal_flip", config["dataset"]["name"] != "mnist")
            if flip:
                ops.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return self.transform(image), label


def _fit_teacher(c, prep, train_data, validation_data, testing_loader, device, directory):
    # Reproducible initialization and data shuffling.
    seed = c["experiment"]["seed"]
    seed_all(seed)
    model = build_model(c["teacher"], c["dataset"]).to(device)
    optimizer = make_optimizer(model, prep["optimizer"])
    schedule = Schedule(optimizer, prep["scheduler"],
                        dict(epochs=prep["epochs"], batch_size_schedule=[prep["batch_size"]]))
    loader = DataLoader(
        train_data, batch_size=prep["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        # Avoid a singleton BatchNorm batch without dropping normal remainders.
        drop_last=len(train_data) % prep["batch_size"] == 1,
    )
    validation = DataLoader(validation_data, batch_size=c["evaluation"]["batch_size"])
    stopping = EarlyStopping(prep.get("early_stopping", {}))
    best_loss, best_epoch, best_state = math.inf, 0, None
    started = time.perf_counter()
    with (directory / "metrics.csv").open("w", newline="") as csv_file, (
        directory / "metrics.jsonl"
    ).open("w") as json_file:
        writer = None
        for epoch in range(1, prep["epochs"] + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            schedule.step(epoch - 1)
            model.train()
            metrics = ClassificationMetrics(c["dataset"]["num_classes"], device)
            objective_sum = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(normalize(images, c["dataset"]))
                loss = F.cross_entropy(logits, labels,
                                       label_smoothing=prep.get("label_smoothing", 0.0))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite teacher loss")
                loss.backward()
                optimizer.step()
                metrics.update(logits, labels)
                objective_sum += loss.item() * len(labels)
            training = metrics.compute()
            train_seconds = time.perf_counter() - epoch_start
            measured = evaluate(model, validation, c["dataset"], device)
            if measured["loss"] < best_loss:
                best_loss, best_epoch = measured["loss"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stopped = stopping.step(measured["loss"], epoch)
            # Reporting only: preserve training RNG state during test evaluation.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                tested = evaluate(model, testing_loader, c["dataset"], device)
            row = {
                "epoch": epoch, "batch_size": prep["batch_size"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_objective": objective_sum / training["samples"],
                "train_seconds": train_seconds,
                "train_samples_per_second": training["samples"] / train_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                        if device.type == "cuda" else None),
                "best_epoch": best_epoch, "best_val_loss": best_loss,
                "early_stopping_bad_epochs": stopping.bad_epochs, "early_stopped": stopped,
            }
            for prefix, values in (("train", training), ("val", measured), ("test", tested)):
                row.update({f"{prefix}_{key}": value for key, value in values.items()
                            if key != "distillation_loss"})
            scalars = {key: value for key, value in row.items() if not isinstance(value, list)}
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(scalars))
                writer.writeheader()
            writer.writerow(scalars)
            csv_file.flush()
            json_file.write(json.dumps(row, allow_nan=False) + "\n")
            json_file.flush()
            print(
                f"Epoch {epoch}/{prep['epochs']} | "
                f"train_loss={training['loss']:.4f} train_acc={training['accuracy']:.2f}% | "
                f"val_loss={measured['loss']:.4f} val_acc={measured['accuracy']:.2f}% | "
                f"test_loss={tested['loss']:.4f} test_acc={tested['accuracy']:.2f}%",
                flush=True,
            )
            if stopped:
                break
    summary = dict(best_epoch=best_epoch, best_val_loss=best_loss,
                   epochs_run=epoch, early_stopped=stopped, teacher_training=prep)
    return summary, best_state


def train_teacher(config):
    """Export the validation-NLL winner; test metrics are reporting-only."""
    c = validate(copy.deepcopy(config))
    prep = c["teacher_training"]
    seed = c["experiment"]["seed"]
    # Ignore obsolete search settings in older configuration files.
    prep.pop("tuning", None)
    _validate_training(prep)
    device = get_device(c)
    data = real_dataset(c, "train", purpose="teacher_training")
    train_indices, validation_indices = stratified_split(
        data.targets, prep.get("validation_fraction", 0.1), seed
    )
    run_config = copy.deepcopy(c)
    run_config["experiment"]["name"] += "_teacher"
    run = make_run(run_config)
    torch.save({"train": train_indices, "validation": validation_indices}, run / "split.pt")
    train_data = TrainingView(Subset(data, train_indices), c)
    validation_data = Subset(data, validation_indices)
    testing_loader = test_loader(c)
    summary, state = _fit_teacher(
        c, prep, train_data, validation_data, testing_loader, device, run
    )
    torch.save(state, run / "best.pt")
    del state
    model = build_model(c["teacher"], c["dataset"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    # Re-evaluate the validation-selected checkpoint for the final report.
    testing = evaluate(model, testing_loader, c["dataset"], device)
    output = Path(c["teacher"]["checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "best.pt", output)
    results = dict(
        selection_metric="validation cross-entropy (NLL)",
        **summary, seed=seed, train_samples=len(train_indices),
        validation_samples=len(validation_indices), test=testing,
        teacher_parameters=parameter_count(model), checkpoint=str(output),
    )
    (run / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(
        f"Selected teacher test | loss={testing['loss']:.4f} accuracy={testing['accuracy']:.2f}%",
        flush=True,
    )
    return output

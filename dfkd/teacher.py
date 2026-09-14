"""Validation-only teacher selection with reproducible, bounded random search."""

import copy
import csv
import itertools
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dfkd.config import set_nested, validate
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


def trial_configs(prep, seed):
    """Baseline first, then distinct grid points sampled without replacement."""
    tuning = prep.get("tuning", {})
    trials = tuning.get("trials", 1)
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("tuning.trials must be a positive integer")
    baseline = copy.deepcopy(prep)
    baseline.pop("tuning", None)
    if trials == 1:
        return [baseline]
    space = tuning.get("search_space", {})
    allowed = {"optimizer.lr", "optimizer.weight_decay", "optimizer.momentum",
               "batch_size", "label_smoothing"}
    if not space or set(space) - allowed:
        raise ValueError(f"Search space must use nonempty subsets of {sorted(allowed)}")
    if any(not isinstance(v, list) or not v for v in space.values()):
        raise ValueError("Search-space values must be nonempty lists")
    size = math.prod(len(v) for v in space.values())
    if size > 10000:
        raise ValueError("Search grid is limited to 10000 combinations")
    candidates, seen = [], {json.dumps(baseline, sort_keys=True)}
    for values in itertools.product(*space.values()):
        candidate = copy.deepcopy(baseline)
        for key, value in zip(space, values):
            set_nested(candidate, key, value)
        signature = json.dumps(candidate, sort_keys=True)
        if signature not in seen:
            candidates.append(candidate)
            seen.add(signature)
    if trials > len(candidates) + 1:
        raise ValueError("tuning.trials exceeds the number of distinct configurations")
    return [baseline] + random.Random(seed).sample(candidates, trials - 1)


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


def _validate_trial(prep):
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


def _train_trial(c, prep, train_data, validation_data, device, directory, trial):
    # Identical initialization and initial shuffle seed for fair comparisons.
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
    directory.mkdir()
    resolved = copy.deepcopy(c)
    resolved["teacher_training"] = prep
    (directory / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
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
                    raise FloatingPointError(f"Non-finite teacher loss in trial {trial}")
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
            row = {
                "trial": trial, "epoch": epoch, "batch_size": prep["batch_size"],
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
            for prefix, values in (("train", training), ("val", measured)):
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
                f"Trial {trial} | Epoch {epoch}/{prep['epochs']} | "
                f"train_loss={training['loss']:.4f} train_acc={training['accuracy']:.2f}% | "
                f"val_loss={measured['loss']:.4f} val_acc={measured['accuracy']:.2f}%",
                flush=True,
            )
            if stopped:
                break
    summary = dict(trial=trial, best_epoch=best_epoch, best_val_loss=best_loss,
                   epochs_run=epoch, early_stopped=stopped, teacher_training=prep)
    (directory / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary, best_state


def train_teacher(config):
    """Export the validation-NLL winner; access the test set only after selection."""
    c = validate(copy.deepcopy(config))
    prep = c["teacher_training"]
    seed = c["experiment"]["seed"]
    trials = trial_configs(prep, seed)
    for trial in trials:
        _validate_trial(trial)
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
    summaries, winner = [], None
    for number, trial in enumerate(trials, 1):
        summary, state = _train_trial(
            c, trial, train_data, validation_data, device, run / f"trial_{number:03d}", number
        )
        summaries.append(summary)
        if winner is None or summary["best_val_loss"] < winner["best_val_loss"]:
            winner = summary
            torch.save(state, run / "best.pt")
        del state
        (run / "trials.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    model = build_model(c["teacher"], c["dataset"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    # No test-set reads or scores have influenced the search or early stopping.
    testing = evaluate(model, test_loader(c), c["dataset"], device)
    output = Path(c["teacher"]["checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "best.pt", output)
    selected = copy.deepcopy(c)
    selected["teacher_training"] = copy.deepcopy(winner["teacher_training"])
    selected["teacher_training"]["tuning"] = {"trials": 1}
    (run / "best_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
    results = dict(
        selection_metric="validation cross-entropy (NLL)", best_trial=winner["trial"],
        best_epoch=winner["best_epoch"], best_val_loss=winner["best_val_loss"],
        trials=len(trials), seed=seed, train_samples=len(train_indices),
        validation_samples=len(validation_indices), test=testing,
        teacher_parameters=parameter_count(model), checkpoint=str(output),
    )
    (run / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(
        f"Test | loss={testing['loss']:.4f} accuracy={testing['accuracy']:.2f}%",
        flush=True,
    )
    return output

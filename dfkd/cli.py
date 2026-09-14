"""Command line entrypoint: dfkd <command> --config CONFIG [overrides]."""

import argparse
import json

import yaml

from dfkd.config import load, set_nested, validate
from dfkd.data import test_loader
from dfkd.models import build_model, parameter_count
from dfkd.train import evaluate, get_device, load_teacher, seed_all, train_student, train_teacher


def resolve(args):
    c = load(args.config)
    for name, key in (
        ("temperature", "distillation.temperature"),
        ("seed", "experiment.seed"),
        ("num_samples", "synthetic_data.num_samples"),
        ("epochs", "teacher_training.epochs" if args.command == "train_teacher" else "training.epochs"),
    ):
        value = getattr(args, name)
        if value is not None:
            set_nested(c, key, value)
    for entry in args.set:
        if "=" not in entry:
            raise SystemExit("--set requires dotted.key=YAML_VALUE")
        key, value = entry.split("=", 1)
        set_nested(c, key, yaml.safe_load(value))
    return validate(c)


def main(argv=None):
    p = argparse.ArgumentParser(description="Data-free KD from random synthetic priors")
    p.add_argument("command", choices=["train_teacher", "distill", "evaluate", "inspect"])
    p.add_argument("--config", required=True)
    p.add_argument("--temperature", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-samples", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--checkpoint", help="student checkpoint, for evaluate")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    c = resolve(args)
    seed_all(c["experiment"]["seed"])
    if args.command == "inspect":
        print(yaml.safe_dump(c, sort_keys=False))
        for role in ("teacher", "student"):
            model = build_model(c[role], c["dataset"])
            print(f"{role}: {c[role]['architecture']} {parameter_count(model):,} parameters")
        return
    if args.command == "train_teacher":
        print(train_teacher(c))
        return
    if args.command == "distill":
        print(train_student(c))
        return
    if not args.checkpoint:
        p.error("evaluate requires --checkpoint STUDENT_CHECKPOINT")
    import torch

    device = get_device(c)
    teacher = load_teacher(c, device)
    student = build_model(c["student"], c["dataset"]).to(device)
    student.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    loader = test_loader(c)
    teacher_accuracy = evaluate(teacher, loader, c["dataset"], device)["accuracy"]
    result = evaluate(
        student, loader, c["dataset"], device, teacher, c["distillation"]["temperature"]
    )
    result["teacher_accuracy"] = teacher_accuracy
    result["accuracy_retention_percent"] = 100 * result["accuracy"] / teacher_accuracy
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

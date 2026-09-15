"""Small-dataset ViT architecture and pipeline regression checks."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dfkd.config import load, validate
from dfkd.models import build_model, freeze, parameter_count
from dfkd.vit_small import LSA, SPT

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset,classes", [("cifar10", 10), ("cifar100", 100)])
def test_vit_configs_and_architecture(dataset, classes):
    torch.set_num_threads(1)
    c = validate(load(ROOT / f"configs/{dataset}_vit.yaml"))
    teacher = build_model(c["teacher"], c["dataset"])
    student = build_model(c["student"], c["dataset"])
    assert len(teacher.transformer.layers) == 8
    assert len(student.transformer.layers) == 4
    assert isinstance(teacher.to_patch_embedding, SPT)
    assert isinstance(teacher.transformer.layers[0][0].fn, LSA)
    assert teacher.to_patch_embedding.to_patch_tokens[1].normalized_shape == (240,)
    assert teacher.pos_embedding.shape == (1, 65, 256)
    assert parameter_count(teacher) > parameter_count(student)
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32)
        for model in (teacher, student):
            logits = model(x)
            assert logits.shape == (2, classes)
            assert torch.isfinite(logits).all()
    freeze(teacher)
    assert not any(p.requires_grad for p in teacher.parameters())


def test_lsa_masks_self_attention_and_learns_temperature():
    torch.set_num_threads(1)
    attention = LSA(dim=16, heads=2, dim_head=8, dropout=0)
    captured = []
    handle = attention.attend.register_forward_hook(
        lambda module, args, output: captured.append(output.detach())
    )
    attention(torch.randn(2, 5, 16)).square().mean().backward()
    handle.remove()
    assert torch.equal(captured[0].diagonal(dim1=-2, dim2=-1), torch.zeros(2, 2, 5))
    assert attention.temperature.grad is not None
    assert torch.isfinite(attention.temperature.grad)


def test_vit_teacher_and_student_pipeline(tmp_path, monkeypatch):
    import dfkd.teacher as teacher_module
    from dfkd.train import train_student

    torch.set_num_threads(1)
    c = load(ROOT / "configs/cifar10_vit.yaml")
    c["experiment"]["output_dir"] = str(tmp_path / "runs")
    c["teacher"]["checkpoint"] = str(tmp_path / "teacher.pt")
    # Small widths keep the CPU smoke test fast; registered depths remain 8 and 4.
    for role in ("teacher", "student"):
        c[role]["kwargs"].update(patch_size=8, dim=16, heads=2, dim_head=8,
                                  mlp_dim=32, dropout=0, emb_dropout=0)
    c["teacher_training"].update(epochs=1, batch_size=4, augmentation={"enabled": False})
    c["training"].update(epochs=1, batch_size_schedule=[4], device="cpu")
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

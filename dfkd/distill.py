"""Distillation objectives on teacher/student logits."""

import torch.nn.functional as F

from dfkd.registry import Registry

LOSSES = Registry()


@LOSSES.register("kl_divergence")
def kl(student, teacher, temperature):
    return (
        F.kl_div(
            F.log_softmax(student.float() / temperature, dim=-1),
            F.softmax(teacher.float() / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


@LOSSES.register("mse_logits")
def mse(student, teacher, temperature):
    return F.mse_loss(student.float(), teacher.float())


def objective(student, teacher, config):
    return LOSSES.resolve(config["loss"])(student, teacher, config["temperature"])

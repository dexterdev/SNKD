"""Streaming multiclass metrics; no predictions or new dependencies retained."""

import torch
from torch.nn import functional as F


class ClassificationMetrics:
    def __init__(self, classes, device, bins=15):
        self.classes = classes
        self.bins = bins
        self.confusion = torch.zeros(classes, classes, device=device, dtype=torch.long)
        self.bin_confidence = torch.zeros(bins, device=device, dtype=torch.float64)
        self.bin_correct = torch.zeros_like(self.bin_confidence)
        # NLL, top-k hits, Brier sum, confidence sum.
        self.sums = torch.zeros(4, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, labels):
        logits = logits.detach().float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite classification logits")
        probabilities = logits.softmax(1)
        confidence, predictions = probabilities.max(1)
        correct = predictions.eq(labels)
        self.confusion += torch.bincount(
            labels * self.classes + predictions, minlength=self.classes ** 2
        ).reshape(self.classes, self.classes)
        indices = (confidence * self.bins).long().clamp(max=self.bins - 1)
        self.bin_confidence.scatter_add_(0, indices, confidence.double())
        self.bin_correct.scatter_add_(0, indices, correct.double())
        self.sums[0] += F.cross_entropy(logits, labels, reduction="sum").double()
        self.sums[1] += logits.topk(min(5, self.classes), dim=1).indices.eq(
            labels[:, None]
        ).any(1).sum()
        self.sums[2] += (
            probabilities.square().sum(1)
            - 2 * probabilities.gather(1, labels[:, None]).squeeze(1) + 1
        ).double().sum()
        self.sums[3] += confidence.double().sum()

    def compute(self):
        cm = self.confusion.cpu().double()
        support = cm.sum(1)
        count = int(support.sum())
        if count == 0:
            raise ValueError("Classification dataset is empty")
        tp = cm.diag()
        precision = tp / cm.sum(0).clamp_min(1)
        recall = tp / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
        sums = self.sums.cpu()
        ece = (self.bin_confidence - self.bin_correct).abs().sum().item() / count
        return {
            "samples": count,
            "loss": sums[0].item() / count,
            "accuracy": 100 * tp.sum().item() / count,
            "top5_accuracy": 100 * sums[1].item() / count,
            "macro_precision": precision.mean().item(),
            "macro_recall": recall.mean().item(),
            "macro_f1": f1.mean().item(),
            "weighted_f1": (f1 * support).sum().item() / count,
            "balanced_accuracy": 100 * recall[support > 0].mean().item(),
            "ece": ece,
            "brier_score": sums[2].item() / count,
            "mean_confidence": sums[3].item() / count,
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.long().tolist(),
            "confusion_matrix": cm.long().tolist(),
        }

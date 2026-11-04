from __future__ import annotations

import math
import numpy as np

from .losses import EPSILON


def validate_threshold(threshold):
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number in [0,1]")
    return float(threshold)


class BinaryMetrics:


    def __init__(self, threshold=0.5, loss="bce"):
        self.threshold = validate_threshold(threshold)
        if loss not in ("bce", "balanced_bce"):
            raise ValueError("loss must be bce or balanced_bce")
        self.loss_name = loss
        self.tp = self.fp = self.fn = self.tn = 0
        self.positive_nll = self.negative_nll = 0.0

    def update(self, labels, probabilities):
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities)
        if labels.shape != probabilities.shape or labels.size == 0:
            raise ValueError("Labels and probabilities must have the same nonempty shape")
        if (labels.dtype.kind not in "fiub" or not np.isfinite(labels).all()
                or not np.logical_or(labels == 0, labels == 1).all()):
            raise ValueError("Labels must be finite binary values 0 or 1")
        if (probabilities.dtype.kind not in "fiu" or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("Probabilities must be finite values in [0,1]")
        truth = labels.astype(bool)
        prediction = probabilities >= self.threshold
        self.tp += int(np.count_nonzero(truth & prediction))
        self.fp += int(np.count_nonzero(~truth & prediction))
        self.fn += int(np.count_nonzero(truth & ~prediction))
        self.tn += int(np.count_nonzero(~truth & ~prediction))
        p = np.clip(probabilities.astype(np.float64), EPSILON, 1.0 - EPSILON)
        self.positive_nll += float(-np.log(p[truth]).sum(dtype=np.float64))
        self.negative_nll += float(-np.log1p(-p[~truth]).sum(dtype=np.float64))

    def result(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn
        if total == 0:
            raise ValueError("No voxels have been evaluated")

        def ratio(a, b):
            return float(a / b) if b else 0.0

        foreground_iou = ratio(tp, tp + fp + fn)
        background_iou = ratio(tn, tn + fp + fn)
        if self.loss_name == "balanced_bce":
            beta = (tn + fp) / total
            loss = (beta * self.positive_nll + (1.0 - beta) * self.negative_nll) / total
        else:
            loss = (self.positive_nll + self.negative_nll) / total
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "voxel_count": total,
                "accuracy": ratio(tp + tn, total), "precision": ratio(tp, tp + fp),
                "recall": ratio(tp, tp + fn), "dice": ratio(2 * tp, 2 * tp + fp + fn),
                "iou": foreground_iou, "miou": (foreground_iou + background_iou) / 2.0,
                "loss": float(loss)}


def compute_metrics(labels, probabilities, threshold=0.5, loss="bce"):
    accumulator = BinaryMetrics(threshold, loss)
    accumulator.update(labels, probabilities)
    return accumulator.result()

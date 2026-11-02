"""Synthetic Python ML Algorithm Development, Calibration & Threshold Optimization Sandbox.

IMPORTANT DISCLOSURE:
This module is a synthetic demonstration created specifically for an Upwork proposal (2117067966830227913).
It operates on in-memory synthetic fixtures only and is NOT a prior commercial client project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    youden_j: float


def compute_brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Compute Brier score (mean squared probability error) for binary classification."""
    if len(y_true) != len(y_prob) or not y_true:
        raise ValueError("y_true and y_prob must be non-empty and have equal length")
    total = 0.0
    for yt, yp in zip(y_true, y_prob):
        if yt not in (0, 1):
            raise ValueError("y_true must contain binary labels 0 or 1")
        if yp < 0.0 or yp > 1.0:
            raise ValueError("y_prob must lie in [0.0, 1.0]")
        total += (yp - float(yt)) ** 2
    return round(total / len(y_true), 6)


def compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 5) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width probability bins."""
    if len(y_true) != len(y_prob) or not y_true or n_bins <= 0:
        raise ValueError("Invalid inputs for compute_ece")
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i for i, p in enumerate(y_prob)
            if (lo <= p < hi) or (b == n_bins - 1 and lo <= p <= hi)
        ]
        if not idxs:
            continue
        avg_prob = sum(y_prob[i] for i in idxs) / len(idxs)
        avg_true = sum(y_true[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_prob - avg_true)
    return round(ece, 6)


def temperature_scale_probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    """Apply post-hoc temperature scaling to binary logits to improve calibration."""
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    scaled: list[float] = []
    for z in logits:
        val = max(-50.0, min(50.0, z / temperature))
        scaled.append(round(1.0 / (1.0 + math.exp(-val)), 6))
    return scaled


def select_optimal_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    grid: Sequence[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
) -> ThresholdMetrics:
    """Select the decision threshold on out-of-fold probabilities that maximizes F1 (tie-breaking on Youden's J)."""
    if len(y_true) != len(y_prob) or not y_true:
        raise ValueError("y_true and y_prob must be non-empty and have equal length")

    best: ThresholdMetrics | None = None
    for thr in grid:
        tp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp >= thr)
        fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= thr)
        fn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp < thr)
        tn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp < thr)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        youden_j = recall + specificity - 1.0

        cand = ThresholdMetrics(
            threshold=round(thr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            youden_j=round(youden_j, 4),
        )
        if best is None or (cand.f1, cand.youden_j) > (best.f1, best.youden_j):
            best = cand

    assert best is not None
    return best

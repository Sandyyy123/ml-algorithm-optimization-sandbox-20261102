"""Unit tests for the synthetic Python ML algorithm development & optimization sandbox."""
from __future__ import annotations

import pytest
from ml_optimization_sandbox import (
    compute_brier_score,
    compute_ece,
    select_optimal_threshold,
    temperature_scale_probabilities,
)


def test_compute_brier_score_perfect_predictions():
    assert compute_brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == 0.0


def test_compute_brier_score_known_value():
    # Errors: 0.1^2, 0.2^2, 0.1^2, 0.2^2 -> mean = (0.01 + 0.04 + 0.01 + 0.04)/4 = 0.025
    assert compute_brier_score([0, 1, 0, 1], [0.1, 0.8, 0.1, 0.8]) == 0.025


def test_compute_brier_score_rejects_invalid_labels_or_probs():
    with pytest.raises(ValueError):
        compute_brier_score([0, 2], [0.2, 0.8])
    with pytest.raises(ValueError):
        compute_brier_score([0, 1], [-0.1, 0.8])


def test_compute_ece_returns_zero_when_perfectly_calibrated():
    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.0, 1.0, 1.0]
    assert compute_ece(y_true, y_prob, n_bins=5) == 0.0


def test_temperature_scale_probabilities_softens_overconfident_logits():
    logits = [-3.0, 0.0, 3.0]
    unscaled = temperature_scale_probabilities(logits, temperature=1.0)
    softened = temperature_scale_probabilities(logits, temperature=2.0)
    assert unscaled[1] == 0.5 and softened[1] == 0.5
    assert softened[2] < unscaled[2]
    assert softened[0] > unscaled[0]


def test_temperature_scale_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="strictly positive"):
        temperature_scale_probabilities([1.0, -1.0], 0.0)


def test_select_optimal_threshold_maximizes_f1_on_imbalanced_fold():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1]
    y_prob = [0.05, 0.10, 0.15, 0.22, 0.28, 0.35, 0.42, 0.85]
    best = select_optimal_threshold(y_true, y_prob)
    assert best.threshold == 0.30
    assert best.f1 == 1.0
    assert best.youden_j == 1.0


def test_select_optimal_threshold_rejects_empty_input():
    with pytest.raises(ValueError):
        select_optimal_threshold([], [])

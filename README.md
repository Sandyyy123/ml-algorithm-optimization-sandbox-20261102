# Python ML Algorithm Development, Calibration & Performance Optimization Sandbox (Synthetic Demo)

> **Important Disclosure:** This repository is a **synthetic Python demonstration** created specifically to illustrate Python machine learning algorithm development, probability calibration, and threshold optimization for an Upwork proposal (`2117067966830227913`). It uses synthetic in-memory fixtures only and is **not** a prior commercial client project.

## What This Synthetic Sandbox Demonstrates

1. **Brier Score & Expected Calibration Error (`compute_brier_score`, `compute_ece`)**: Quantifies both discrimination and probability calibration across equal-width bins.
2. **Post-Hoc Temperature Scaling (`temperature_scale_probabilities`)**: Softens overconfident logits on out-of-fold validation splits.
3. **Out-of-Fold Decision Threshold Optimization (`select_optimal_threshold`)**: Selects the operating threshold maximizing F1 and Youden's J statistic on imbalanced datasets.

## Run Unit Tests

```bash
pytest -v
```

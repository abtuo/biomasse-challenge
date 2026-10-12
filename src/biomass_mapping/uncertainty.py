from __future__ import annotations

import numpy as np


def conformal_radius(y_true, y_pred, alpha: float = 0.10) -> float:
    scores = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = min(q, 1.0)
    return float(np.quantile(scores, q, method="higher"))

def interval_from_radius(pred, radius: float):
    pred = np.asarray(pred)
    lo = np.maximum(0.0, pred - radius)
    hi = pred + radius
    return lo, hi

def empirical_coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def split_conformal_evaluate(calibration_y, calibration_pred, test_y, test_pred, alpha=0.10):
    """Calibrate on one set and evaluate on a distinct test set."""
    radius = conformal_radius(calibration_y, calibration_pred, alpha=alpha)
    lo, hi = interval_from_radius(test_pred, radius)
    return {
        "radius": radius,
        "coverage": empirical_coverage(test_y, lo, hi),
        "mean_interval_width": float(np.mean(hi - lo)),
    }, lo, hi

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "nrmse_mean": float(mean_squared_error(y_true, y_pred) ** 0.5 / np.mean(y_true)),
    }

def make_rf(params: dict) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
        max_features=params.get("max_features", 0.7),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=int(params.get("random_state", 42)),
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )

def make_ridge(params: dict | None = None) -> Pipeline:
    params = params or {}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=float(params.get("alpha", 1.0)))),
    ])

def make_hist_gradient_boosting(params: dict | None = None) -> Pipeline:
    params = params or {}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.08)),
            max_iter=int(params.get("max_iter", 180)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(params.get("min_samples_leaf", 30)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            random_state=int(params.get("random_state", 42)),
        )),
    ])

def make_model(name: str, config: dict):
    if name == "ridge":
        return make_ridge(config.get("ridge", {}))
    if name == "random_forest":
        return make_rf(config.get("random_forest", config))
    if name == "hist_gradient_boosting":
        return make_hist_gradient_boosting(config.get("hist_gradient_boosting", {}))
    raise ValueError(f"Unknown model: {name}")

def spatial_cv(model, X, y, groups, n_splits=5, max_train_samples=None, random_state=42):
    cv = GroupKFold(n_splits=n_splits)
    fold_rows = []
    oof = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        n_train_available = len(tr)
        if max_train_samples and len(tr) > max_train_samples:
            rng = np.random.default_rng(random_state + fold)
            tr = np.sort(rng.choice(tr, size=max_train_samples, replace=False))
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        row = {"fold": fold}
        row.update(regression_metrics(y.iloc[te], pred))
        row["n_train"] = len(tr)
        row["n_train_available"] = n_train_available
        row["n_test"] = len(te)
        fold_rows.append(row)

    overall = regression_metrics(y, oof)
    return pd.DataFrame(fold_rows), oof, overall

def fixed_fold_cv(model, X, y, fold_ids, max_train_samples=None, random_state=42):
    """Cross-validate on a persisted fold assignment shared by all feature sets."""
    fold_ids = np.asarray(fold_ids)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        tr, te = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        n_train_available = len(tr)
        if max_train_samples and len(tr) > max_train_samples:
            tr = np.sort(np.random.default_rng(random_state + int(fold)).choice(
                tr, size=max_train_samples, replace=False
            ))
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        pred = fitted.predict(X.iloc[te])
        oof[te] = pred
        fold_rows.append({
            "fold": int(fold), **regression_metrics(y.iloc[te], pred),
            "n_train": len(tr), "n_train_available": n_train_available, "n_test": len(te),
        })
    return pd.DataFrame(fold_rows), oof, regression_metrics(y, oof)

def random_holdout(model, X, y, test_size=0.2, random_state=42, max_train_samples=None):
    tr, te = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
    )
    if max_train_samples and len(tr) > max_train_samples:
        rng = np.random.default_rng(random_state)
        tr = np.sort(rng.choice(tr, size=max_train_samples, replace=False))
    m = clone(model)
    m.fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    out = regression_metrics(y.iloc[te], pred)
    out.update(n_train=len(tr), n_test=len(te))
    return out

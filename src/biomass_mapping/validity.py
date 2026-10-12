from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNDomainValidity:
    def __init__(self, n_neighbors: int = 10, threshold_quantile: float = 0.95,
                 max_reference_samples: int = 20000, random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.threshold_quantile = threshold_quantile
        self.max_reference_samples = max_reference_samples
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        if len(X) > self.max_reference_samples:
            reference_idx = rng.choice(len(X), self.max_reference_samples, replace=False)
            reference = X[reference_idx]
        else:
            reference = X
        self.imputer_ = SimpleImputer(strategy="median")
        reference = self.imputer_.fit_transform(reference)
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(reference)
        k = min(self.n_neighbors + 1, len(Z))
        self.nn_ = NearestNeighbors(n_neighbors=k)
        self.nn_.fit(Z)
        d, _ = self.nn_.kneighbors(Z)
        # Exclude the zero-distance self neighbour for the reference threshold.
        self.train_distance_ = d[:, 1:].mean(axis=1) if d.shape[1] > 1 else d[:, 0]
        self.threshold_ = float(np.quantile(self.train_distance_, self.threshold_quantile))
        return self

    def score_samples(self, X):
        Z = self.scaler_.transform(self.imputer_.transform(np.asarray(X)))
        d, _ = self.nn_.kneighbors(Z)
        return d[:, :min(self.n_neighbors, d.shape[1])].mean(axis=1)

    def in_domain(self, X):
        return self.score_samples(X) <= self.threshold_

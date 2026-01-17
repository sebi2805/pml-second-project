from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

DEFAULT_PERPLEXITIES = (5.0, 10.0, 30.0, 50.0)
DEFAULT_LEARNING_RATES = ("auto", 100.0, 200.0)
DEFAULT_MAX_ITERS = (1000,)
DEFAULT_METRICS = ("euclidean",)
DEFAULT_INITS = ("pca", "random")


@dataclass(frozen=True)
class TSNEResult:
    params: dict[str, object]
    embedding: np.ndarray
    trustworthiness: float


def _prepare_features(features: np.ndarray, standardize: bool) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array (n_samples, n_features).")
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")
    if standardize:
        return StandardScaler().fit_transform(features)
    return features


def run_tsne(
    features: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    *,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    init: str = "pca",
    random_state: int = 42,
    standardize: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    x = _prepare_features(features, standardize)
    if perplexity >= x.shape[0]:
        raise ValueError("perplexity must be < n_samples.")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
    )
    coords = tsne.fit_transform(x)
    if labels is None:
        return coords, None
    return coords, np.asarray(labels)


def grid_search_tsne(
    features: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    *,
    param_grid: Optional[dict[str, Iterable[object]]] = None,
    random_state: int = 42,
    standardize: bool = True,
    n_neighbors: int = 5,
) -> tuple[list[TSNEResult], Optional[np.ndarray]]:
    x = _prepare_features(features, standardize)
    y = np.asarray(labels) if labels is not None else None

    if param_grid is None:
        param_grid = {
            "perplexity": DEFAULT_PERPLEXITIES,
            "learning_rate": DEFAULT_LEARNING_RATES,
            "max_iter": DEFAULT_MAX_ITERS,
            "metric": DEFAULT_METRICS,
            "init": DEFAULT_INITS,
        }

    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    results: list[TSNEResult] = []

    for idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo, strict=False))
        perplexity = float(params.get("perplexity", 30.0))
        if perplexity >= x.shape[0]:
            continue

        metric = str(params.get("metric", "euclidean"))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=params.get("learning_rate", "auto"),
            max_iter=int(params.get("max_iter", 1000)),
            metric=metric,
            init=str(params.get("init", "pca")),
            random_state=random_state + idx,
        )
        coords = tsne.fit_transform(x)
        k = min(n_neighbors, x.shape[0] - 1)
        score = trustworthiness(x, coords, n_neighbors=k, metric=metric)
        results.append(TSNEResult(params=params, embedding=coords, trustworthiness=score))

    results.sort(key=lambda item: item.trustworthiness, reverse=True)
    return results, y

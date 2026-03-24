"""
Dimensionality reduction module.

Provides t-SNE, PCA, and UMAP reduction for speech representations,
with optional pre-PCA for t-SNE speedup.
"""

import warnings
from typing import List, Optional

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import config


def reduce_features(
    features: np.ndarray,
    method: str = "tsne",
    n_components: int = 3,
    random_state: int = None,
    pre_pca_dim: Optional[int] = None,
) -> np.ndarray:
    """Reduce feature dimensionality using the specified method.

    Args:
        features: Input features of shape (N, D).
        method: Reduction method: 'tsne', 'pca', or 'umap'.
        n_components: Target dimensionality.
        random_state: Random seed. Defaults to config.
        pre_pca_dim: If set, apply PCA to this dimension before t-SNE/UMAP.

    Returns:
        Reduced features of shape (N, n_components).
    """
    if random_state is None:
        random_state = config.TSNE_RANDOM_STATE

    if pre_pca_dim is None and method == "tsne":
        pre_pca_dim = config.TSNE_PRE_PCA_DIM

    X = features.astype(np.float32)

    # Optional pre-PCA
    if pre_pca_dim is not None and X.shape[1] > pre_pca_dim:
        pdim = min(pre_pca_dim, X.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = PCA(n_components=pdim, random_state=random_state).fit_transform(X)

    if method == "pca":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PCA(n_components=n_components,
                       random_state=random_state).fit_transform(X)

    elif method == "tsne":
        perplexity = _safe_perplexity(X.shape[0])
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate=config.TSNE_LEARNING_RATE,
            n_iter=config.TSNE_N_ITER,
        )
        Y = tsne.fit_transform(X)
        return _zscore(Y)

    elif method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=config.UMAP_N_NEIGHBORS,
            min_dist=config.UMAP_MIN_DIST,
            random_state=config.UMAP_RANDOM_STATE,
        )
        return reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def reduce_sentence_features(
    sentence_data: List[List[np.ndarray]],
    method: str = "tsne",
    n_components: int = 3,
    random_state: int = None,
    pre_pca_dim: Optional[int] = None,
) -> List[List[np.ndarray]]:
    """Reduce features while preserving sentence structure.

    Concatenates all segments, reduces jointly, then splits back.

    Args:
        sentence_data: Nested list [sentence][audio_file] of (T, D) arrays.
        method: Reduction method.
        n_components: Target dimensions.
        random_state: Random seed.
        pre_pca_dim: Pre-PCA dimension.

    Returns:
        Same nested structure with (T, n_components) arrays.
    """
    # Flatten
    segments = [seg for sent in sentence_data for seg in sent]
    if not segments:
        return sentence_data

    seg_lengths = [s.shape[0] for s in segments]
    flat = np.concatenate(segments, axis=0)

    reduced = reduce_features(flat, method=method,
                              n_components=n_components,
                              random_state=random_state,
                              pre_pca_dim=pre_pca_dim)

    # Split back
    splits = []
    idx = 0
    for length in seg_lengths:
        splits.append(reduced[idx:idx + length])
        idx += length

    result = [[] for _ in range(len(sentence_data))]
    k = 0
    for i, sent in enumerate(sentence_data):
        for _ in sent:
            result[i].append(splits[k])
            k += 1

    return result


def _safe_perplexity(n_samples: int) -> int:
    """Compute a safe perplexity value for t-SNE."""
    if n_samples <= 10:
        return 5
    return int(max(5, min(30, (n_samples - 1) // 3)))


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize along axis 0."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)

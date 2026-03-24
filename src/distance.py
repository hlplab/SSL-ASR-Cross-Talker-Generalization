"""
Distance and similarity computation module.

Implements DTW-based distance measures with weighted Minkowski metrics,
used for comparing speech feature sequences.
"""

from typing import List, Optional

import numpy as np
from numba import njit


@njit
def weighted_minkowski(vec1: np.ndarray, vec2: np.ndarray,
                       tau: float = 2.0, w: float = 1.0) -> float:
    """Compute weighted Minkowski distance between two vectors.

    Args:
        vec1, vec2: Input vectors of same dimensionality.
        tau: Minkowski order (1=Manhattan, 2=Euclidean, etc.).
        w: Per-dimension weight.

    Returns:
        Weighted Minkowski distance.
    """
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total ** (1.0 / tau)


@njit
def dtw_distance(seq1: np.ndarray, seq2: np.ndarray,
                 tau: float = 2.0) -> float:
    """Compute normalized Dynamic Time Warping distance.

    Uses weighted Minkowski as the local cost function.

    Args:
        seq1, seq2: Sequences of shape (T1, D) and (T2, D).
        tau: Minkowski order for local distance.

    Returns:
        Normalized DTW distance (divided by mean sequence length).
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = weighted_minkowski(seq1[i - 1], seq2[j - 1], tau)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],       # insertion
                dtw_matrix[i, j - 1],       # deletion
                dtw_matrix[i - 1, j - 1]    # match
            )

    return dtw_matrix[n, m] / ((n + m) / 2.0)


@njit
def dtw_similarity(seq1: np.ndarray, seq2: np.ndarray,
                   tau: float = 2.0, k: float = 0.05) -> float:
    """Compute DTW-based similarity (exponential transform of distance).

    similarity = exp(-normalized_dtw_distance * k)

    Args:
        seq1, seq2: Sequences of shape (T1, D) and (T2, D).
        tau: Minkowski order.
        k: Scaling parameter controlling similarity falloff.

    Returns:
        Similarity score in (0, 1].
    """
    dist = dtw_distance(seq1, seq2, tau)
    return np.exp(-dist * k)


def compute_pairwise_distances(
    sequences: List[np.ndarray],
    tau: float = 2.0,
) -> np.ndarray:
    """Compute pairwise DTW distance matrix for a list of sequences.

    Args:
        sequences: List of (T, D) arrays.
        tau: Minkowski order.

    Returns:
        Distance matrix of shape (N, N).
    """
    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(sequences[i], sequences[j], tau)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def compute_mean_pairwise_euclidean(
    mean_representations: List[np.ndarray],
) -> float:
    """Compute mean pairwise Euclidean distance between mean representations.

    Args:
        mean_representations: List of (D,) vectors.

    Returns:
        Mean pairwise Euclidean distance.
    """
    if len(mean_representations) < 2:
        return 0.0
    distances = []
    for i in range(len(mean_representations) - 1):
        for j in range(i + 1, len(mean_representations)):
            distances.append(np.linalg.norm(
                mean_representations[i] - mean_representations[j]))
    return np.mean(distances)

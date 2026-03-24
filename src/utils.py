"""
Utility module.

Provides plotting helpers, result formatting, and misc convenience functions.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


# =============================================================================
# Serialization
# =============================================================================

def save_pickle(obj, path: str):
    """Save object to a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Data Helpers
# =============================================================================

def flatten_features(feature_dict: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Flatten a nested feature dict into a 2D array.

    Args:
        feature_dict: {speaker: {word: array(T, D)}}.

    Returns:
        Array of shape (N, D).
    """
    return np.array([
        frame
        for speaker_words in feature_dict.values()
        for word_frames in speaker_words.values()
        for frame in word_frames
    ])


def unflatten_features(
    original_dict: Dict,
    reduced_array: np.ndarray,
) -> Dict:
    """Reconstruct nested dict structure from a flat reduced array.

    Args:
        original_dict: Template dict (same structure as output of extract functions).
        reduced_array: Flat reduced features of shape (N, D_reduced).

    Returns:
        Nested dict with same keys but reduced-dimensionality arrays.
    """
    import copy
    result = {}
    count = 0
    for speaker, words in original_dict.items():
        result[speaker] = {}
        for word, frames in words.items():
            T = frames.shape[1] if frames.ndim == 3 else frames.shape[0]
            result[speaker][word] = reduced_array[count:count + T]
            count += T
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_layer_z_scores(
    results_df: pd.DataFrame,
    title: str = "GLMM Z-value by Model Layer",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """Plot z-values across model layers.

    Args:
        results_df: DataFrame with 'layer' and 'z_value' (or 'train_z_value') columns.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    z_col = 'z_value' if 'z_value' in results_df.columns else 'train_z_value'

    fig, ax = plt.subplots(figsize=figsize)

    mean_z = results_df.groupby('layer')[z_col].mean()
    std_z = results_df.groupby('layer')[z_col].std()

    layers = mean_z.index.astype(str)
    ax.bar(range(len(layers)), mean_z.values, yerr=std_z.values,
           capsize=3, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_xlabel("Model Layer")
    ax.set_ylabel(f"Mean {z_col}")
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_optimization_history(
    study,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Plot Optuna optimization history.

    Args:
        study: Optuna study object.
        save_path: Optional save path.
        figsize: Figure size.
    """
    try:
        from optuna.visualization.matplotlib import plot_optimization_history as _plot
        fig = _plot(study)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    except ImportError:
        print("optuna[visualization] not installed. Skipping plot.")


# =============================================================================
# Format helpers
# =============================================================================

def format_results_table(
    results_df: pd.DataFrame,
    sort_col: str = 'z_value',
    ascending: bool = False,
) -> pd.DataFrame:
    """Format results for display, sorting by the specified column.

    Args:
        results_df: Results DataFrame.
        sort_col: Column to sort by.
        ascending: Sort direction.

    Returns:
        Formatted DataFrame.
    """
    df = results_df.copy()
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)
    return df

"""
Variability analysis module.

Implements 9 variability metrics for measuring phonetic dispersion in
speech representations, as described in the paper. Each metric captures
a different aspect of variability in the training talkers' productions.
"""

import itertools
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

import config


def compute_variability(
    human_df,
    feature_dict: Dict,
    tau: float = 2.0,
    k: float = 0.05,
    method: str = "VariabilityAcrossTime",
    set_group: str = "set1,set2",
) -> Dict:
    """Compute a variability metric for each unique training talker set.

    Args:
        human_df: Human results DataFrame.
        feature_dict: Dict mapping talker_id -> list of sentence feature lists.
        tau: Minkowski order.
        k: Similarity scaling parameter.
        method: One of VARIABILITY_METHODS.
        set_group: "set1,set2" or "set2,set1" to select sentence sets.

    Returns:
        Dict mapping talker_set_key -> variability value.
    """
    method_map = {
        "VariabilityAcrossTime": variability_across_time,
        "VariabilityInSimilarityAcrossWords": variability_similarity_across_words,
        "VariabilityInSimilarityAcrossPhoneme": variability_similarity_across_phoneme,
        "VariabilityWithinPhonemeCategories": variability_within_phoneme_categories,
        "VariabilityBetweenPhonemeCategory": variability_between_phoneme_category,
        "VariabilityCoefficient": variability_coefficient,
        "VariabilitySpectralEntropy": variability_spectral_entropy,
        "VariabilityMeanPairwiseDistance": variability_mean_pairwise_distance,
        "VariabilityTemporalGradient": variability_temporal_gradient,
    }

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

    return method_map[method](human_df, feature_dict, tau, k, set_group)


# =============================================================================
# Method Implementations
# =============================================================================

def _get_training_sets(human_df):
    """Extract unique sorted training talker sets from the DataFrame."""
    unique_ids = (human_df["TrainingTalkerID"]
                  .apply(lambda x: ",".join(sorted(x.split(", "))))
                  .unique())
    return [s.split(",") for s in unique_ids]


def _collect_features(feature_dict, talker_set, set_group: str):
    """Collect features for a set of talkers, selecting the appropriate sentence subset."""
    feature_list = []
    for talker in talker_set:
        if talker in feature_dict:
            all_feats = list(itertools.chain(*feature_dict[talker]))
            if set_group == "set1,set2":
                feature_list += all_feats[:54]
            elif set_group == "set2,set1":
                feature_list += all_feats[54:]
    return feature_list


def variability_across_time(human_df, feature_dict, tau, k, set_group):
    """Mean frame-level dispersion across time.

    For each word segment, computes the Minkowski dispersion of frames
    from their mean, then averages across all segments.
    """
    from src.distance import weighted_minkowski

    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        var_val = np.mean([
            np.sum(np.abs(f - np.mean(f, axis=0)) ** 2, axis=1) ** (1.0 / tau)
            for f in features
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] > 0
        ]) if features else 0.0
        result["_".join(talker_set)] = np.exp(-var_val * k)
    return result


def variability_similarity_across_words(human_df, feature_dict, tau, k, set_group):
    """1 - mean pairwise similarity between word-level mean representations.

    Measures how distinct different words are from each other in the
    training set.
    """
    from src.distance import weighted_minkowski

    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        mean_features = [np.mean(f, axis=0) for f in features
                         if isinstance(f, np.ndarray) and f.ndim == 2]
        sims = []
        for i, s1 in enumerate(mean_features[:-1]):
            for s2 in mean_features[i + 1:]:
                sims.append(np.exp(-weighted_minkowski(s1, s2, tau) * k))
        result["_".join(talker_set)] = 1 - np.mean(sims) if sims else 0.0
    return result


def variability_similarity_across_phoneme(human_df, feature_dict, tau, k, set_group):
    """1 - mean pairwise similarity between phoneme-level mean representations.

    Groups features by phoneme label, computes mean per phoneme, then
    measures pairwise similarity.
    """
    from src.distance import weighted_minkowski

    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        phone_dict = defaultdict(list)
        for f in features:
            if isinstance(f, dict):
                for key, v in f.items():
                    phone_dict[key].append(v)

        means = [np.mean([np.mean(v, axis=0) for v in vals], axis=0)
                 for key, vals in phone_dict.items() if len(vals) >= 2]
        sims = []
        for i, s1 in enumerate(means[:-1]):
            for s2 in means[i + 1:]:
                sims.append(np.exp(-weighted_minkowski(s1, s2, tau) * k))
        result["_".join(talker_set)] = 1 - np.mean(sims) if sims else 0.0
    return result


def variability_within_phoneme_categories(human_df, feature_dict, tau, k, set_group):
    """Within-phoneme-category variance, averaged across categories.

    For each phoneme category, computes the variance of individual tokens
    around the category mean, then averages across categories.
    Higher values = more within-category variability.
    """
    training_sets = _get_training_sets(human_df)
    result = {}

    # Compute per-talker within-phoneme variance
    all_talkers = set(j for t in training_sets for j in t)
    talker_var = {}
    for talker in all_talkers:
        if talker not in feature_dict:
            continue
        features = list(itertools.chain(*feature_dict[talker]))
        if set_group == "set2,set1":
            features = features[54:]

        phone_dict = {}
        for f in features:
            if isinstance(f, dict):
                for key, v in f.items():
                    if key not in phone_dict:
                        phone_dict[key] = []
                    phone_dict[key].extend(v if isinstance(v, list) else [v])

        phone_vars = []
        for key, value in phone_dict.items():
            v_arr = np.array([np.mean(v) for v in value]) if value else []
            if len(v_arr) > 1:
                grand_mean = np.mean(v_arr)
                phone_vars.append(np.mean((v_arr - grand_mean) ** 2))
        talker_var[talker] = np.mean(phone_vars) if phone_vars else 0.0

    for talker_set in training_sets:
        vals = [talker_var.get(t, 0.0) for t in talker_set]
        result["_".join(talker_set)] = np.exp((1 - np.mean(vals)) * k)
    return result


def variability_between_phoneme_category(human_df, feature_dict, tau, k, set_group):
    """Between-category dispersion: variance of phoneme category means.

    Computes mean representation per phoneme category, then measures
    the variance of these category means.
    """
    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        phone_dict = defaultdict(list)
        for f in features:
            if isinstance(f, dict):
                for key, v in f.items():
                    phone_dict[key].append(v)

        if len(phone_dict) >= 2:
            cat_means = []
            for key, vals in phone_dict.items():
                concat = np.concatenate([np.mean(v, axis=0, keepdims=True)
                                        for v in vals], axis=0)
                cat_means.append(np.mean(concat, axis=0))
            cat_stack = np.stack(cat_means)
            between_var = np.mean(np.var(cat_stack, axis=0))
        else:
            between_var = 0.0
        result["_".join(talker_set)] = np.exp(-between_var * k)
    return result


def variability_coefficient(human_df, feature_dict, tau, k, set_group):
    """Coefficient of variation (std/mean) across feature dimensions.

    Scale-invariant measure of dispersion.
    """
    eps = 1e-8
    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        cv_vals = []
        for f in features:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] > 1:
                cv_vals.append(np.mean(
                    np.std(f, axis=0) / (np.abs(np.mean(f, axis=0)) + eps)))
        var_val = np.mean(cv_vals) if cv_vals else 0.0
        result["_".join(talker_set)] = np.exp(-var_val * k)
    return result


def variability_spectral_entropy(human_df, feature_dict, tau, k, set_group):
    """Shannon entropy of the feature value distribution.

    Captures the spread of feature values by computing per-dimension
    entropy, then averaging.
    """
    eps = 1e-10
    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        all_feats = [f for f in features
                     if isinstance(f, np.ndarray) and f.ndim == 2]
        if all_feats:
            concat = np.concatenate(all_feats, axis=0)
            entropies = []
            for d in range(concat.shape[1]):
                col = concat[:, d]
                hist, _ = np.histogram(col, bins=50, density=True)
                hist = hist / (hist.sum() + eps)
                ent = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + eps))
                entropies.append(ent)
            var_val = np.mean(entropies)
        else:
            var_val = 0.0
        result["_".join(talker_set)] = np.exp(-var_val * k)
    return result


def variability_mean_pairwise_distance(human_df, feature_dict, tau, k, set_group):
    """Mean pairwise Euclidean distance between word-level mean representations.

    Captures overall spread of the word space.
    """
    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        mean_reps = [np.mean(f, axis=0) for f in features
                     if isinstance(f, np.ndarray) and f.ndim == 2]
        if len(mean_reps) >= 2:
            distances = []
            for i in range(len(mean_reps) - 1):
                for j in range(i + 1, len(mean_reps)):
                    distances.append(np.linalg.norm(mean_reps[i] - mean_reps[j]))
            var_val = np.mean(distances)
        else:
            var_val = 0.0
        result["_".join(talker_set)] = np.exp(-var_val * k)
    return result


def variability_temporal_gradient(human_df, feature_dict, tau, k, set_group):
    """Mean norm of frame-to-frame differences.

    Captures temporal dynamics — how rapidly features change over time.
    """
    training_sets = _get_training_sets(human_df)
    result = {}
    for talker_set in training_sets:
        features = _collect_features(feature_dict, talker_set, set_group)
        grad_vals = []
        for f in features:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] > 1:
                diffs = np.diff(f, axis=0)
                grad_vals.append(np.mean(np.linalg.norm(diffs, axis=1)))
        var_val = np.mean(grad_vals) if grad_vals else 0.0
        result["_".join(talker_set)] = np.exp(-var_val * k)
    return result

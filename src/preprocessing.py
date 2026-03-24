"""
Preprocessing module.

Handles TextGrid parsing, word-level alignment, keyword extraction,
fold splitting for cross-validation, and dataset-specific preprocessing.
"""

import os
import copy
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config


def load_human_data(path: str = None) -> pd.DataFrame:
    """Load and preprocess the human experimental results.

    Args:
        path: Path to the Excel file. Defaults to config.HUMAN_DATA_PATH.

    Returns:
        DataFrame with human experimental data.
    """
    if path is None:
        path = str(config.HUMAN_DATA_PATH)

    df = pd.read_excel(path)

    # Filter for Experiment 1a
    if "Experiment" in df.columns:
        df_1a = df[df["Experiment"] == "1a"].copy()
    else:
        df_1a = df.copy()

    # Normalize TrainingTalkerID sorting
    if "TrainingTalkerID" in df_1a.columns:
        df_1a["TrainingTalkerID1"] = (
            df_1a["TrainingTalkerID"]
            .astype(str)
            .apply(lambda x: ",".join(sorted(x.split(", "))) if pd.notna(x) else x)
        )

    return df_1a


def get_keywords_dict(df: pd.DataFrame) -> Dict:
    """Build a mapping from SentenceID to list of keywords.

    Args:
        df: Human results DataFrame with 'SentenceID' and 'Keyword' columns.

    Returns:
        Dict mapping sentenceID -> [keyword1, keyword2, ...].
    """
    keywords_dict = {}
    for row in df.values:
        sentenceID = row[df.columns.get_loc("SentenceID")]
        keyword = row[df.columns.get_loc("Keyword")]
        if sentenceID not in keywords_dict:
            keywords_dict[sentenceID] = []
        if keyword not in keywords_dict[sentenceID]:
            keywords_dict[sentenceID].append(keyword)
    return dict(sorted(keywords_dict.items()))


def get_keywords_list(df: pd.DataFrame) -> Dict:
    """Same as get_keywords_dict but with different key access pattern."""
    return get_keywords_dict(df)


def get_training_paths(training_talker_id: str) -> List[str]:
    """Convert raw TrainingTalkerID string to internal talker identifiers.

    Converts identifiers like 'ENG001' to 'ALL_001_M_ENG' format.

    Args:
        training_talker_id: Comma-separated string of talker IDs.

    Returns:
        List of internal talker identifier strings.
    """
    talker_ids = []
    for each_id in training_talker_id.split(", "):
        if each_id[:3] == "CMN":
            talker_ids.append(f"ALL_{each_id[-3:]}_M_CMN")
        else:
            talker_ids.append(f"ALL_{each_id[-3:]}_M_ENG")
    return talker_ids


def create_word_features(
    audio_dir: str,
    df: pd.DataFrame,
    reduced_data: List[List[np.ndarray]],
) -> Tuple[List[List[np.ndarray]], Dict]:
    """Align reduced features to word-level units using TextGrid annotations.

    For each audio file, parses the TextGrid to find word boundaries within
    each sentence, then extracts the corresponding feature frames from the
    reduced (e.g., t-SNE) representations.

    Args:
        audio_dir: Directory with .wav/.TextGrid pairs.
        df: Human results DataFrame.
        reduced_data: Reduced features indexed as [sentence_idx][audio_idx],
                      each element is an array of shape (T, D).

    Returns:
        Tuple of:
            - word_features: List of keyword feature lists.
            - feature_dict: Dict mapping talker_id -> list of sentence feature lists.
    """
    import textgrid

    keywords_dict = get_keywords_dict(df)
    keywords = [j for i in list(keywords_dict.values()) for j in i]
    audio_paths = _get_audio_paths(audio_dir)

    out_dict = {}
    word_features = [[] for _ in range(len(keywords))]

    for audio_idx, audio_path in enumerate(audio_paths):
        current_talker = os.path.basename(audio_path)[:13]
        if current_talker not in out_dict:
            out_dict[current_talker] = [[] for _ in range(32)]

        tg = textgrid.TextGrid.fromFile(audio_path[:-3] + "TextGrid")
        tg_sentence = _parse_sentences(tg, config.ALL_INDICES)
        tg_word = [i for i in tg[1] if i.mark != "" and i.mark != "sp"]

        count = 0
        for sent_idx, each_sentence in enumerate(tg_sentence):
            sentence_length = each_sentence.maxTime - each_sentence.minTime

            for key_word in list(keywords_dict.values())[sent_idx]:
                start, end = None, None
                for each_word_tg in tg_word:
                    if (each_word_tg.mark.lower() == key_word and
                            each_word_tg.minTime >= each_sentence.minTime and
                            each_word_tg.maxTime <= each_sentence.maxTime):
                        start = each_word_tg.minTime
                        end = each_word_tg.maxTime
                        break

                word_start_ratio = (start - each_sentence.minTime) / sentence_length
                word_end_ratio = (end - each_sentence.minTime) / sentence_length
                n_frames = reduced_data[sent_idx][audio_idx].shape[0]
                word_start = round(n_frames * word_start_ratio)
                word_end = round(n_frames * word_end_ratio)

                features = copy.deepcopy(
                    reduced_data[sent_idx][audio_idx][word_start:word_end, :])
                word_features[count].append(features)
                out_dict[current_talker][sent_idx].append(features)
                count += 1

    return word_features, out_dict


def create_word_features_phoneme(
    audio_dir: str,
    df: pd.DataFrame,
    reduced_data: List[List[np.ndarray]],
) -> Dict:
    """Create phoneme-level features using TextGrid tier 2.

    Similar to create_word_features but operates at the phoneme level,
    returning nested dicts mapping phoneme -> features.

    Args:
        audio_dir: Directory with .wav/.TextGrid pairs.
        df: Human results DataFrame.
        reduced_data: Reduced features indexed as [sentence_idx][audio_idx].

    Returns:
        Dict mapping talker_id -> list of sentence data (list of phoneme dicts).
    """
    import textgrid

    keywords_dict = get_keywords_dict(df)
    audio_paths = _get_audio_paths(audio_dir)
    out_dict = {}

    for audio_idx, audio_path in enumerate(audio_paths):
        current_talker = os.path.basename(audio_path)[:13]
        if current_talker not in out_dict:
            out_dict[current_talker] = [[] for _ in range(32)]

        tg = textgrid.TextGrid.fromFile(audio_path[:-3] + "TextGrid")
        tg_sentence = _parse_sentences(tg, config.ALL_INDICES)
        tg_word = [i for i in tg[1] if i.mark != "" and i.mark != "sp"]
        tg_phoneme = [i for i in tg[2] if i.mark != "" and i.mark != "sp"]

        for sent_idx, each_sentence in enumerate(tg_sentence):
            sentence_length = each_sentence.maxTime - each_sentence.minTime

            for key_word in list(keywords_dict.values())[sent_idx]:
                start, end = None, None
                for each_word_tg in tg_word:
                    if (each_word_tg.mark.lower() == key_word and
                            each_word_tg.minTime >= each_sentence.minTime and
                            each_word_tg.maxTime <= each_sentence.maxTime):
                        start = each_word_tg.minTime
                        end = each_word_tg.maxTime
                        break

                phone_key_dict = {}
                for each_phoneme_tg in tg_phoneme:
                    if (each_phoneme_tg.minTime >= start and
                            each_phoneme_tg.maxTime <= end):
                        p_start = int(reduced_data[sent_idx][audio_idx].shape[0] *
                                      (each_phoneme_tg.minTime - each_sentence.minTime) /
                                      sentence_length)
                        p_end = int(reduced_data[sent_idx][audio_idx].shape[0] *
                                    (each_phoneme_tg.maxTime - each_sentence.minTime) /
                                    sentence_length) + 1
                        features = copy.deepcopy(
                            reduced_data[sent_idx][audio_idx][p_start:p_end, :])
                        phone_key_dict[each_phoneme_tg.mark] = features

                out_dict[current_talker][sent_idx] = [phone_key_dict]

    return out_dict


def assign_folds(df: pd.DataFrame, n_folds: int = 3,
                 random_state: int = 42) -> pd.DataFrame:
    """Assign cross-validation folds using stratified splitting.

    Groups participants by their TrainingTestSet/Condition/TestTalker
    combination, then splits groups into folds.

    Args:
        df: DataFrame with 'WorkerID', 'TrainingTestSet', 'Condition2',
            'TestTalkerID' columns.
        n_folds: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with added 'fold' column.
    """
    participants = (df[['WorkerID', 'TrainingTestSet', 'Condition2', 'TestTalkerID']]
                    .drop_duplicates().reset_index(drop=True))

    participants['combined_key'] = (
        participants['TrainingTestSet'].astype(str) + "_" +
        participants['Condition2'].astype(str) + "_" +
        participants['TestTalkerID'].astype(str)
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    participants['fold'] = -1

    for fold_idx, (_, test_index) in enumerate(
            skf.split(participants, participants['combined_key'])):
        participants.loc[test_index, 'fold'] = fold_idx + 1

    return df.merge(participants[['WorkerID', 'fold']], on='WorkerID', how='left')


def standardize_features(sentences: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """Z-score standardize features globally across all sentences.

    Args:
        sentences: Nested list of (T, D) arrays.

    Returns:
        Standardized nested list with same structure.
    """
    all_frames = [frame for sent in sentences for seg in sent for frame in seg]
    stacked = np.asarray(all_frames, dtype=np.float32)
    mu = stacked.mean(axis=0, keepdims=True)
    sd = stacked.std(axis=0, keepdims=True) + 1e-8
    standardized = (stacked - mu) / sd

    count = 0
    filled = [[] for _ in range(len(sentences))]
    for i, sent in enumerate(sentences):
        new_segs = []
        for seg in sent:
            T = seg.shape[0]
            new_segs.append(standardized[count:count + T])
            count += T
        filled[i] = new_segs

    return filled


# =============================================================================
# Internal Helpers
# =============================================================================

def _get_audio_paths(audio_dir: str) -> List[str]:
    """Recursively find .wav files, returned in reversed order."""
    paths = []
    for dirpath, _, filenames in os.walk(audio_dir):
        for f in sorted(filenames):
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(dirpath, f))
    return paths[::-1]


def _parse_sentences(tg, set_list: List[int]):
    """Parse TextGrid sentence tier with boundary correction."""
    tg_sentence = list(tg[0])
    for idx, itv in enumerate(tg_sentence):
        if itv.mark != "" and idx > 0:
            tg_sentence[idx - 1].maxTime = itv.minTime
    tg_sentence = [itv for itv in tg_sentence if itv.mark != ""]
    tg_sentence = [tg_sentence[i] for i in set_list]
    return tg_sentence

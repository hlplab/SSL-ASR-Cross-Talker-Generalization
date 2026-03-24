"""
Feature extraction module.

Extracts speech representations from HuBERT (and compatible models) at both
the CNN feature encoder and Transformer encoder levels.
"""

import os
import gc
import copy
import warnings
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import librosa
import tqdm

import config


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str = None):
    """Load the pre-trained HuBERT model and processor.

    Args:
        model_name: HuggingFace model identifier. Defaults to config.MODEL_NAME.

    Returns:
        Tuple of (model, processor, device).
    """
    from transformers import AutoProcessor, AutoModelForCTC

    if model_name is None:
        model_name = config.MODEL_NAME

    device = get_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    model.to(device).eval()

    return model, processor, device


# =============================================================================
# CNN Layer Extraction
# =============================================================================

class CNNActivationCatcher:
    """Captures intermediate activations from HuBERT's CNN feature encoder.

    Registers forward hooks on each convolutional layer in the feature
    extractor to capture per-layer outputs during inference.

    Args:
        feature_extractor: The model's feature_extractor module.
        select_rule: Optional callable(name, module) -> bool to filter layers.
    """

    def __init__(self, feature_extractor: nn.Module,
                 select_rule: Optional[Callable] = None):
        self.feat = feature_extractor
        self.handles = []
        self.cache = {}
        self.names = []

        for name, module in self.feat.named_modules():
            if select_rule is not None and not select_rule(name, module):
                continue
            if isinstance(module, nn.LayerNorm):
                self.names.append(name)
                self.handles.append(
                    module.register_forward_hook(self._hook_factory(name)))
            elif isinstance(module, nn.Conv1d) and select_rule is None:
                # Default: hook on Conv1d if no LayerNorm filter
                self.names.append(name)
                self.handles.append(
                    module.register_forward_hook(self._hook_factory(name)))

    def _hook_factory(self, name: str):
        def _hook(module, inp, out):
            with torch.no_grad():
                self.cache[name] = out.detach().cpu()
        return _hook

    def clear(self):
        self.cache.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def ordered_names(self) -> List[str]:
        return self.names

    def get_outputs(self) -> Dict[str, torch.Tensor]:
        return self.cache


def extract_cnn_layers(
    audio_dir: str,
    set_list: List[int],
    model,
    processor,
    layer_indices: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, List[List[np.ndarray]]]:
    """Extract CNN feature encoder outputs for each sentence segment.

    Args:
        audio_dir: Directory containing .wav files and paired .TextGrid files.
        set_list: List of sentence indices to extract (from TextGrid tier 0).
        model: Pre-loaded HuBERT model.
        processor: Pre-loaded processor matching the model.
        layer_indices: Which CNN layers to extract (0-6). None = all layers.
        cache_dir: Optional directory to save/load intermediate results.
        device: Torch device for computation.

    Returns:
        Dict mapping layer_index -> list of sentences, where each sentence
        is a list of arrays of shape (T, C) (one per audio file).
    """
    if device is None:
        device = get_device()

    forward_module = getattr(model, "hubert", model)
    feat_extractor = forward_module.feature_extractor

    # Register hooks on Conv1d layers
    conv_layers = feat_extractor.conv_layers
    if layer_indices is None:
        layer_indices = list(range(len(conv_layers)))

    activations = {}
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            x = out if not isinstance(out, (list, tuple)) else out[0]
            if x.dim() == 3 and x.shape[1] < x.shape[2]:
                x = x.transpose(1, 2)
            activations[idx] = x.detach().cpu().squeeze(0).numpy()
        return hook

    for idx in layer_indices:
        if hasattr(conv_layers[idx], "conv"):
            target = conv_layers[idx].conv
        else:
            target = conv_layers[idx]
        handles.append(target.register_forward_hook(make_hook(idx)))

    # Collect audio files
    audio_paths = _get_audio_paths(audio_dir)

    # Initialize sentence containers
    layer_sentences = {L: [[] for _ in range(len(set_list))] for L in layer_indices}

    for audio_path in tqdm.tqdm(audio_paths, desc="Extracting CNN features"):
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        wave = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        tg_path = os.path.splitext(audio_path)[0] + ".TextGrid"
        import textgrid
        tg = textgrid.TextGrid.fromFile(tg_path)
        tg_sentence = _parse_textgrid_sentences(tg, set_list)

        for s_idx, seg in enumerate(tg_sentence):
            beg = int(seg.minTime * config.SAMPLE_RATE)
            end = int(seg.maxTime * config.SAMPLE_RATE)
            seg_wave = wave[beg:end]

            if seg_wave.size == 0:
                for L in layer_indices:
                    layer_sentences[L][s_idx].append(np.zeros((0, 1)))
                continue

            inputs = processor(seg_wave, sampling_rate=config.SAMPLE_RATE,
                               return_tensors="pt").input_values.to(device)
            activations.clear()

            with torch.no_grad():
                _ = forward_module(inputs, output_hidden_states=False,
                                   return_dict=True)

            for L in layer_indices:
                arr = activations.get(L)
                if arr is not None:
                    layer_sentences[L][s_idx].append(arr.astype(np.float32))
                else:
                    layer_sentences[L][s_idx].append(np.zeros((0, 1)))

            del inputs
            gc.collect()

    for h in handles:
        h.remove()

    return layer_sentences


# =============================================================================
# Transformer Layer Extraction
# =============================================================================

def extract_transformer_layers(
    audio_dir: str,
    model,
    processor,
    layer_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, np.ndarray]:
    """Extract Transformer encoder hidden states.

    For datasets with isolated word-level audio files (e.g., Nygaard),
    extracts the specified transformer layers for each audio file.

    Args:
        audio_dir: Directory containing .wav files.
        model: Pre-loaded HuBERT model.
        processor: Pre-loaded processor.
        layer_indices: Which transformer layers to extract. None = [1] (first TR layer).
        device: Torch device.

    Returns:
        Dict mapping layer_index -> dict mapping filename -> hidden states (1, T, D).
    """
    if device is None:
        device = get_device()
    if layer_indices is None:
        layer_indices = [1]

    forward_module = getattr(model, "hubert", model)
    model.eval()

    audio_paths = _get_audio_paths(audio_dir)
    out_dict = {L: {} for L in layer_indices}

    for audio_path in tqdm.tqdm(audio_paths, desc="Extracting Transformer features"):
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        wave = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        basename = os.path.basename(audio_path)
        inputs = processor(wave, sampling_rate=config.SAMPLE_RATE,
                           return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            outputs = forward_module(inputs, output_hidden_states=True)
            for L in layer_indices:
                out_dict[L][basename] = copy.deepcopy(
                    outputs.hidden_states[L].cpu().numpy())

        torch.cuda.empty_cache()

    return out_dict


def build_sentence_features(
    audio_dir: str,
    set_list: List[int],
    model,
    processor,
    layer_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, List[List[np.ndarray]]]:
    """Extract Transformer hidden states organized by sentence segments.

    Similar to extract_cnn_layers but for Transformer layers.
    Parses TextGrid to segment audio into sentences and extracts features
    for each segment.

    Args:
        audio_dir: Directory with .wav/.TextGrid pairs.
        set_list: Sentence indices to extract.
        model, processor: Pre-loaded model/processor.
        layer_indices: Transformer layers (0-24 for HuBERT-Large).
        device: Torch device.

    Returns:
        Dict mapping layer_index -> list of sentences (each sentence is a list
        of (T, D) arrays, one per audio file).
    """
    if device is None:
        device = get_device()
    if layer_indices is None:
        layer_indices = config.TRANSFORMER_LAYERS

    forward_module = getattr(model, "hubert", model)
    model.eval()

    audio_paths = _get_audio_paths(audio_dir)
    layer_sentences = {L: [[] for _ in range(len(set_list))] for L in layer_indices}

    for audio_path in tqdm.tqdm(audio_paths, desc="Extracting Transformer sentence features"):
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        wave = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        import textgrid
        tg = textgrid.TextGrid.fromFile(
            os.path.splitext(audio_path)[0] + ".TextGrid")
        tg_sentence = _parse_textgrid_sentences(tg, set_list)

        for s_idx, seg in enumerate(tg_sentence):
            beg = int(seg.minTime * config.SAMPLE_RATE)
            end = int(seg.maxTime * config.SAMPLE_RATE)
            seg_wave = wave[beg:end]

            if seg_wave.size == 0:
                for L in layer_indices:
                    layer_sentences[L][s_idx].append(np.zeros((0, 1024)))
                continue

            inputs = processor(seg_wave, sampling_rate=config.SAMPLE_RATE,
                               return_tensors="pt").input_values.to(device)

            with torch.no_grad():
                outputs = forward_module(inputs, output_hidden_states=True)
                for L in layer_indices:
                    hidden = outputs.hidden_states[L].cpu().squeeze(0).numpy()
                    layer_sentences[L][s_idx].append(hidden.astype(np.float32))

            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return layer_sentences


# =============================================================================
# Helpers
# =============================================================================

def _get_audio_paths(audio_dir: str) -> List[str]:
    """Recursively find all .wav files in a directory."""
    paths = []
    for dirpath, _, filenames in os.walk(audio_dir):
        for f in sorted(filenames):
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(dirpath, f))
    return paths[::-1]  # Match original code's reversed ordering


def _parse_textgrid_sentences(tg, set_list: List[int]):
    """Parse TextGrid to extract sentence intervals based on set_list indices.

    Modifies interval boundaries so that empty intervals are removed and
    sentence boundaries align with word onsets.

    Args:
        tg: A textgrid.TextGrid object.
        set_list: Indices of sentences to extract from tier 0.

    Returns:
        List of textgrid.Interval objects for the selected sentences.
    """
    import textgrid
    tg_sentence = list(tg[0])

    # Fix boundaries: push end of empty intervals to next interval's start
    for idx, itv in enumerate(tg_sentence):
        if itv.mark != "":
            if idx > 0:
                tg_sentence[idx - 1].maxTime = itv.minTime

    tg_sentence = [itv for itv in tg_sentence if itv.mark != ""]
    tg_sentence = [tg_sentence[i] for i in set_list]
    return tg_sentence

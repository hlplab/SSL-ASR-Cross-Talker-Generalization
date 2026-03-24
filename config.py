"""
Configuration for Cross-Talker Generalization analysis pipeline.

All configurable parameters are centralized here. Modify this file to adapt
the pipeline to different datasets or model configurations.
"""

from pathlib import Path
import os

# =============================================================================
# Paths (modify these to point to your data)
# =============================================================================

# Base directory for the project
BASE_DIR = Path(__file__).parent

# Directory containing audio files (.wav) with paired TextGrid annotations
AUDIO_DIR = BASE_DIR / "data" / "speech_files"

# Human experimental results (Excel format)
# Expected columns: Experiment, SentenceID, Keyword, TrainingTalkerID,
#                   TestTalkerID, Filename, IsCorrect, Condition2, TrainingTestSet, WorkerID
HUMAN_DATA_PATH = BASE_DIR / "data" / "test.xlsx"

# Nygaard dataset path (optional, for Study 2)
NYGAARD_DATA_PATH = BASE_DIR / "data" / "mixed_training_scored_data.xlsx"
NYGAARD_AUDIO_DIR = BASE_DIR / "data" / "Nygaard_audio"

# Bradlow dataset paths (optional, for Study 3)
BRADLOW_DATA_PATH = BASE_DIR / "data" / "bradlow_data.xlsx"
BRADLOW_AUDIO_DIR = BASE_DIR / "data" / "bradlow_audio"

# Cache directory for intermediate results
CACHE_DIR = BASE_DIR / "results"

# =============================================================================
# Model Configuration
# =============================================================================

# Pre-trained model identifier (HuBERT-Large fine-tuned on LibriSpeech)
MODEL_NAME = "facebook/hubert-large-ls960-ft"

# Target sample rate for audio processing
SAMPLE_RATE = 16000

# =============================================================================
# Sentence Selection Indices
# =============================================================================

# Sentence indices used in the experiments
# Set 1: Exposure set sentences; Set 2: Generalization set sentences
SET1_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
SET2_INDICES = [17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 37, 40]
ALL_INDICES = SET1_INDICES + SET2_INDICES

# =============================================================================
# Feature Extraction
# =============================================================================

# Transformer layer indices to extract (0=first hidden state after CNN)
# For HuBERT-Large: 0-24 (25 transformer layers)
TRANSFORMER_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# CNN layer indices to extract (HuBERT has 7 CNN layers: 0-6)
CNN_LAYERS = [0, 1, 2, 3, 4, 5, 6]

# =============================================================================
# Dimensionality Reduction
# =============================================================================

# Default reduction method: 'tsne', 'pca', 'umap'
DEFAULT_REDUCTION_METHOD = "tsne"

# Target dimensionality after reduction
REDUCED_DIM = 3

# t-SNE parameters
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = "auto"
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_PRE_PCA_DIM = 50  # PCA dimension before t-SNE (speedup)

# PCA parameters
PCA_RANDOM_STATE = 42

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# =============================================================================
# Distance / Similarity
# =============================================================================

# Minkowski parameter (tau) for weighted distance
DEFAULT_TAU = 2

# Default similarity scaling parameter k
DEFAULT_K = 0.05

# =============================================================================
# GLMM Configuration
# =============================================================================

# GLMM formula template for similarity analysis
# Variables: similarity_scaled, numCorrect, numIncorrect
GLMM_FORMULA_SIMILARITY = (
    "cbind(numCorrect, numIncorrect) ~ 1 + similarity_scaled + "
    "(1 | SentenceID / Keyword) + (1 | TestTalkerID)"
)

# GLMM formula template for variability analysis
GLMM_FORMULA_VARIABILITY = (
    "cbind(numCorrect, numIncorrect) ~ 1 + variability_scaled + "
    "(1 | SentenceID / Keyword) + (1 | TestTalkerID)"
)

# GLMM optimizer settings
GLMM_OPTIMIZER = "bobyqa"
GLMM_MAXFUN = 100000

# =============================================================================
# Cross-Validation
# =============================================================================

# Number of folds for cross-validation
N_FOLDS = 3
CV_RANDOM_STATE = 42

# =============================================================================
# Variability Analysis
# =============================================================================

# Available variability methods
VARIABILITY_METHODS = [
    "VariabilityAcrossTime",
    "VariabilityInSimilarityAcrossWords",
    "VariabilityInSimilarityAcrossPhoneme",
    "VariabilityWithinPhonemeCategories",
    "VariabilityBetweenPhonemeCategory",
    "VariabilityCoefficient",
    "VariabilitySpectralEntropy",
    "VariabilityMeanPairwiseDistance",
    "VariabilityTemporalGradient",
]

# L2 regularization weight for k optimization
L2_ALPHA = 0.1

# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================

# Number of Optuna trials
N_OPTUNA_TRIALS = 200

# k search range for Optuna
K_SEARCH_MIN = 0.001
K_SEARCH_MAX = 10.0
K_SEARCH_STEP = 0.001

# =============================================================================
# R Environment (for rpy2)
# =============================================================================

# Set these to match your R installation
# On Windows, typically: C:\Program Files\R\R-4.x.x
R_HOME = os.environ.get("R_HOME", r"C:\Program Files\R\R-4.4.1")
R_BIN_PATH = os.path.join(R_HOME, "bin", "x64")

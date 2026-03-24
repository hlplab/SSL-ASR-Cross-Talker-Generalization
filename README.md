# Cross-Talker Generalization

> **Latent speech representations learned through self-supervised learning predict listeners' generalization of adaptation across talkers**

**Authors:** Zhengyang Jin, Yuhao Zhu, T. Florian Jaeger

📋 **Accepted at:** CogSci 2025 (Annual Meeting of the Cognitive Science Society)

📄 **Paper:** [Link TBD - preprint coming soon]

📄 **Proceedings:** *Proceedings of the Annual Meeting of the Cognitive Science Society, 47*.

> **Citation:**
> ```
> Jin, Z., Zhu, Y., & Jaeger, T. F. (2025). Latent speech representations learned 
> through self-supervised learning predict listeners' generalization of adaptation 
> across talkers. Proceedings of the Annual Meeting of the Cognitive Science 
> Society, 47.
> ```

🔗 **Repositories:**
- [Main Repo](https://github.com/dashpulsar/Cross-Talker-Generalization) (this repo)
- [Lab Repo](https://github.com/hlplab/SSL-ASR-Cross-Talker-Generalization)

---

## Overview

This repository implements an **ASR-based exemplar model** to investigate how human listeners adapt to and generalize non-native (L2) accented speech. Inspired by exemplar theory, we leverage **self-supervised speech representations** (HuBERT) to model perceptual similarity between talkers and predict human transcription accuracy.

**Key idea:** If HuBERT's latent representations encode talker-independent phonetic information, then the acoustic similarity between training and test talkers (as measured in HuBERT space) should predict listeners' accuracy in perceiving accented speech from novel talkers.

Our approach extends previous work by:
- **Utilizing ASR-derived latent representations** to quantify talker similarity.
- **Applying dynamic time warping (DTW) and t-SNE** for perceptual trajectory alignment.
- **Predicting listener adaptation using mixed-effects logistic regression**.

## Background

Human listeners adapt quickly to novel talkers and accents, yet the underlying mechanisms remain unclear. Exemplar theory suggests that speech perception relies on rich, stored perceptual traces. However, past research has focused on **highly controlled phonetic contrasts**, leaving open the question of whether **these mechanisms also explain generalization in natural speech.**

This project **bridges that gap** by:
- Using **self-supervised ASR models ([HuBERT](https://arxiv.org/abs/2106.07447))** to derive a **latent perceptual space**.
- Measuring **similarity between exposure and test talkers** via **word-level feature distances**.
- Comparing model predictions to **human transcription data from [Xie et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34370501/)**.

## Key Features

- **ASR-Based Perceptual Space**  
  - Extracts **512-dimensional embeddings** from HuBERT’s CNN/Transformer layers.
  - Applies dimensionality reduction (t-SNE/PCA/UMAP) to project high-dimensional representations into **latent spaces**.

- **Word-Level Perceptual Similarity**  
  - Aligns speech trajectories via **DTW**.
  - Computes similarity using optimizable **exponential distance function**.

- **Generalization Prediction**  
  - Models listener adaptation using **mixed-effects logistic regression** (GLMM).
  - Evaluates the influence of **talker exposure conditions** on generalization.

## Methodology

### 1. Data
We use **Experiment 1a from [Xie et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34370501/)** ([OSF link](https://osf.io/brwx5/)), where 320 participants transcribed sentences from University of Northwestern [**Archive of L1 and L2 Scripted and Spontaneous Transcripts And Recordings**](https://speechbox.linguistics.northwestern.edu/allsstar) Mandarin-accented English talkers (Bradlow, A. R. (n.d.) ALLSSTAR). Listeners were exposed to:
- **Control Condition:** Native (L1) English talkers.
- **Multi-Talker Condition:** Different L2 talkers.
- **Single-Talker Condition:** One repeated L2 talker.
- **Talker-Specific Condition:** The same L2 talker as in the test phase.

### 2. Similarity Computation
- Align **keyword trajectories** across talkers using **DTW**.
- Compute perceptual similarity using:

$$
  D(i,j) = \sqrt[\tau]{\sum_m w_m|v_{m,i}-v_{m,j}|^\tau}
$$

$$
  \text{similarity} = \exp\left(\frac{-D(w_x, w_y)^k}{|\pi_{\min}|}\right)
$$

### 3. Modeling Human Perception
- Fit **mixed-effects logistic regression** to predict listener transcription accuracy (binary correct/incorrect), with random intercepts for words and talkers.
- Compare the influence of **talker similarity vs. exposure condition**.
- Use 3-fold cross-validation with separate train/validation/test splits and L2 regularization to prevent overfitting.

## Results

- **Similarity-based inference significantly predicts transcription accuracy**.
- **Transformer-derived features outperform CNN-based features**.
- **Generalization effects align with human experimental results**.

---

## Installation

### Requirements

- Python 3.9+
- R 4.x with `lme4` package
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/dashpulsar/Cross-Talker-Generalization.git
cd Cross-Talker-Generalization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Python dependencies
pip install -r requirements.txt

# Install R package (run in R)
# install.packages("lme4")
```

### R Configuration

Set `R_HOME` and `R_BIN_PATH` in `config.py` to match your R installation. On Windows:

```python
R_HOME = r"C:\Program Files\R\R-4.4.1"
R_BIN_PATH = r"C:\Program Files\R\R-4.4.1\bin\x64"
```

On Linux/macOS, rpy2 usually auto-detects R, but you may need to set:

```bash
export R_HOME=/usr/lib/R
```

---

## Project Structure

```text
├── config.py                    # All configurable parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
│
├── src/                         # Core Python modules
│   ├── __init__.py
│   ├── feature_extraction.py    # HuBERT feature extraction (CNN + Transformer)
│   ├── preprocessing.py         # TextGrid parsing, word alignment, fold splitting
│   ├── dimensionality_reduction.py  # t-SNE, PCA, UMAP
│   ├── distance.py              # DTW, Minkowski distance, similarity computation
│   ├── variability.py           # 9 variability metrics
│   ├── glmm.py                  # GLMM fitting via rpy2/lme4
│   └── utils.py                 # Plotting, serialization, helpers
│
├── notebooks/                   # Jupyter notebooks demonstrating the pipeline
│   ├── 01_feature_extraction.ipynb
│   ├── 02_similarity_analysis.ipynb
│   └── 03_variability_analysis.ipynb
│
├── data/                        # Data directory (not tracked in git)
│   └── .gitkeep
│
└── results/                     # Output directory for cached/intermediate results
    └── .gitkeep
```

---

## Pipeline

The analysis follows this pipeline:

### Step 1: Feature Extraction (`01_feature_extraction.ipynb`)
- Load HuBERT-Large fine-tuned model (`facebook/hubert-large-ls960-ft`)
- Extract features from 7 CNN layers and up to 25 Transformer layers
- Apply dimensionality reduction (t-SNE by default)
- Save features as pickle files

### Step 2: Similarity Analysis (`02_similarity_analysis.ipynb`)
- Load extracted features and human experimental data
- Parse TextGrid annotations for word-level alignment
- Compute DTW distances between training/test talker pairs
- Optimize similarity scaling parameter k via Optuna
- Run 3-fold cross-validation with GLMMs
- Visualize z-values across model layers

### Step 3: Variability Analysis (`03_variability_analysis.ipynb`)
- Compute 9 variability metrics for training talker sets
- For each method, optimize k on validation fold
- Evaluate on held-out test fold
- Compare which variability metrics best predict generalization

---

## Data

Due to ethical and consent considerations, raw audio files and human experimental data are **not included** in this repository. To reproduce the results:

1. Place your audio files in `data/speech_files/` (with paired `.TextGrid` files)
2. Place the human results Excel file at `data/test.xlsx`
3. Update paths in `config.py` if needed

### Expected Data Format

**Human results (`test.xlsx`):**
| Column | Description |
|--------|-------------|
| Experiment | "1a" |
| WorkerID | Participant ID |
| SentenceID | Sentence identifier |
| Keyword | Target word |
| TrainingTalkerID | Comma-separated training talker IDs |
| TestTalkerID | Test talker ID |
| Filename | Audio filename |
| IsCorrect | 1/0 accuracy |
| Condition2 | Experimental condition |
| TrainingTestSet | Which word set was used for training |

**Audio files:**
- `.wav` files paired with `.TextGrid` files (same basename)
- TextGrid should have 3 tiers: sentences (tier 0), words (tier 1), phonemes (tier 2)

---

## Variability Methods

| Method | Description |
|--------|-------------|
| `VariabilityAcrossTime` | Mean frame-level temporal dispersion within word segments |
| `VariabilityInSimilarityAcrossWords` | 1 - mean pairwise similarity between word-level means |
| `VariabilityInSimilarityAcrossPhoneme` | 1 - mean pairwise similarity between phoneme-level means |
| `VariabilityWithinPhonemeCategories` | Average within-phoneme-category variance |
| `VariabilityBetweenPhonemeCategory` | Variance across phoneme category means |
| `VariabilityCoefficient` | Mean coefficient of variation (std/mean) per word |
| `VariabilitySpectralEntropy` | Shannon entropy of feature distribution per dimension |
| `VariabilityMeanPairwiseDistance` | Mean pairwise Euclidean distance between word means |
| `VariabilityTemporalGradient` | Mean frame-to-frame difference magnitude |

---

## Configuration

All parameters are centralized in `config.py`:

- **Model:** `MODEL_NAME` (HuBERT variant)
- **Layers:** `TRANSFORMER_LAYERS`, `CNN_LAYERS`
- **Reduction:** `DEFAULT_REDUCTION_METHOD`, `REDUCED_DIM`, t-SNE/PCA/UMAP params
- **Distance:** `DEFAULT_TAU` (Minkowski order), `DEFAULT_K`
- **GLMM:** Formulas, optimizer settings
- **CV:** `N_FOLDS`, `CV_RANDOM_STATE`
- **Optuna:** `N_OPTUNA_TRIALS`, k search range
- **Variability:** Method list, L2 regularization weight

---

## License

This code is provided for research purposes. Please cite the paper if you use this code in your work.

---

## Acknowledgments

- HuBERT model by [Hugging Face Transformers](https://huggingface.co/facebook/hubert-large-ls960-ft)
- GLMM fitting via [lme4](https://cran.r-project.org/package=lme4) and [rpy2](https://rpy2.github.io/)
- Hyperparameter optimization via [Optuna](https://optuna.org/)

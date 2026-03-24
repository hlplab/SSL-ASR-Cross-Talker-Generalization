"""
GLMM fitting module via rpy2/lme4.

Provides functions for fitting Generalized Linear Mixed Models using R's
lme4 package through rpy2, with support for cross-validation and
hyperparameter optimization.
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config

# =============================================================================
# R Environment Setup
# =============================================================================

def setup_r():
    """Initialize R environment and load required packages.

    Call this once at the start of your script/notebook before any
    GLMM operations. Sets R_HOME and PATH, then activates rpy2.
    """
    os.environ['R_HOME'] = config.R_HOME
    os.environ["PATH"] += os.pathsep + config.R_BIN_PATH

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    # Register the safe GLMER helper in R
    ro.r(_R_GLMER_HELPER)

    return ro


def _ensure_r():
    """Lazy R setup, called automatically."""
    try:
        import rpy2.robjects as ro
        return ro
    except Exception:
        return setup_r()


# =============================================================================
# R Helper Functions (embedded R code)
# =============================================================================

_R_GLMER_HELPER = r'''
run_glmer_safe <- function(formula, data, maxfun_val) {
    warn_msgs <- character(0)
    ctrl <- lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = maxfun_val)
    )

    result <- tryCatch({
        withCallingHandlers({
            model <- lme4::glmer(
                formula, data = data, control = ctrl,
                family = stats::binomial(link = "logit")
            )
            summ <- summary(model)
            coefs <- summ$coefficients
            z_val <- coefs[2, 3]
            bic_val <- BIC(model)
            list(
                status = "success",
                z_value = z_val,
                bic = bic_val,
                warnings = warn_msgs,
                error_msg = NA
            )
        }, warning = function(w) {
            warn_msgs <<- c(warn_msgs, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
    }, error = function(e) {
        list(
            status = "error",
            z_value = NA,
            bic = NA,
            warnings = warn_msgs,
            error_msg = conditionMessage(e)
        )
    })
    return(result)
}
'''


# =============================================================================
# Core GLMM Interface
# =============================================================================

def fit_glmm(
    df_model: pd.DataFrame,
    formula: str = None,
    maxfun: int = None,
    verbose: bool = False,
) -> Dict:
    """Fit a GLMM using R's lme4::glmer via rpy2.

    Args:
        df_model: DataFrame with columns matching the formula.
            Expected: numCorrect, numIncorrect, similarity (or variability).
        formula: R formula string. Defaults to config.
        maxfun: Max function evaluations for optimizer. Defaults to config.
        verbose: Print warnings if True.

    Returns:
        Dict with keys: 'status', 'z_value', 'bic', 'warnings', 'error_msg'.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, Formula

    _ensure_r()
    pandas2ri.activate()

    if formula is None:
        formula = config.GLMML_FORMULA_SIMILARITY
    if maxfun is None:
        maxfun = config.GLMML_MAXFUN

    # Ensure required columns exist
    if 'numCorrect' not in df_model.columns:
        if 'IsCorrect' in df_model.columns:
            df_model = df_model.copy()
            df_model['numCorrect'] = (df_model['IsCorrect'] == 1).astype(int)
            df_model['numIncorrect'] = (df_model['IsCorrect'] == 0).astype(int)
        else:
            raise ValueError("DataFrame must have 'numCorrect'/'numIncorrect' or 'IsCorrect' columns")

    r_formula = Formula(formula)
    r_data = pandas2ri.py2rpy(df_model)
    run_glmer = ro.globalenv['run_glmer_safe']

    r_result = run_glmer(r_formula, r_data, maxfun)

    status = str(r_result.rx2('status')[0])
    z_value = float(r_result.rx2('z_value')[0])
    bic = float(r_result.rx2('bic')[0])

    warnings_r = r_result.rx2('warnings')
    warning_str = ""
    if warnings_r is not None and len(warnings_r) > 0:
        warning_str = "; ".join([str(w) for w in warnings_r])

    error_msg = r_result.rx2('error_msg')
    error_str = str(error_msg[0]) if error_msg is not None and not pd.isna(error_msg[0]) else ""

    if verbose and warning_str:
        print(f"GLMM warnings: {warning_str}")

    return {
        'status': status,
        'z_value': z_value,
        'bic': bic,
        'warnings': warning_str,
        'error_msg': error_str,
    }


# =============================================================================
# Data Preparation for GLMM
# =============================================================================

def prepare_similarity_data(
    df_long: pd.DataFrame,
    k: float,
    similarity_col: str = 'raw_distance',
) -> pd.DataFrame:
    """Convert raw distances to similarity and aggregate for GLMM.

    Steps:
    1. Convert distance to similarity: similarity = exp(-distance * k)
    2. Average across files per (Subject, Keyword, ...)
    3. Aggregate correct/incorrect counts for GLMM cbind()

    Args:
        df_long: Long-form DataFrame with per-trial raw distances.
        k: Scaling parameter.
        similarity_col: Column containing raw distance values.

    Returns:
        Aggregated DataFrame ready for GLMM fitting.
    """
    df = df_long.copy()
    df['similarity'] = np.exp(-df[similarity_col] * k)

    # Per-file aggregation
    group_cols = ['Subject', 'FileName', 'Keyword', 'Speaker_full',
                  'TestAccent', 'TrainingTalkerID']
    existing_cols = [c for c in group_cols if c in df.columns]

    df_per_file = df.groupby(existing_cols, as_index=False).agg(
        similarity=('similarity', 'mean'),
        IsCorrect=('IsCorrect', 'first'),
    )

    # Final aggregation
    model_cols = ['Keyword', 'TestAccent', 'TrainingTalkerID',
                  'Speaker_full', 'Subject']
    existing_model = [c for c in model_cols if c in df_per_file.columns]

    df_model = df_per_file.groupby(existing_model, as_index=False).agg(
        similarity=('similarity', 'mean'),
        numCorrect=('IsCorrect', 'sum'),
        numIncorrect=('IsCorrect', lambda x: (x == 0).sum()),
    )

    return df_model


def prepare_variability_data(
    df_var: pd.DataFrame,
    train_mean: float,
    train_sd: float,
) -> pd.DataFrame:
    """Scale variability values and aggregate for GLMM.

    Uses training set statistics to scale both train and test data.

    Args:
        df_var: DataFrame with per-trial variability values.
        train_mean: Mean variability from training set.
        train_sd: Std variability from training set.

    Returns:
        Aggregated DataFrame with variability_scaled column.
    """
    df = df_var.copy()

    if 'TrainingTalkerID1' not in df.columns and 'TrainingTalkerID' in df.columns:
        df['TrainingTalkerID1'] = df['TrainingTalkerID'].apply(
            lambda x: ",".join(sorted(x.split(", "))))

    agg_cols = ['Keyword', 'Condition2', 'TrainingTalkerID1',
                'TestTalkerID', 'SentenceID']
    existing_agg = [c for c in agg_cols if c in df.columns]

    df_model = df.groupby(existing_agg, as_index=False).agg(
        IsCorrect=('IsCorrect', 'mean'),
        variability=('variability', 'mean'),
        numCorrect=('IsCorrect', lambda x: (x == 1).sum()),
        numIncorrect=('IsCorrect', lambda x: (x == 0).sum()),
    )

    if train_sd == 0:
        return None
    df_model['variability_scaled'] = (df_model['variability'] - train_mean) / (2 * train_sd)

    return df_model


# =============================================================================
# Cross-Validation GLMM
# =============================================================================

def cross_validate_glmm(
    df_long: pd.DataFrame,
    df_folds: pd.DataFrame,
    k: float,
    n_folds: int = 3,
    formula: str = None,
    tau: float = 2.0,
) -> pd.DataFrame:
    """Run k-fold cross-validation with fixed k parameter.

    Args:
        df_long: Precomputed distance DataFrame with 'Fold' column.
        df_folds: Full DataFrame with fold assignments.
        k: Fixed scaling parameter.
        n_folds: Number of folds.
        formula: GLMM formula.
        tau: Not used directly (distances already precomputed).

    Returns:
        DataFrame with per-fold z-values and BIC.
    """
    results = []
    for hold_out_fold in range(n_folds):
        train_df = df_long[df_long['Fold'] != hold_out_fold].copy()

        if train_df.empty:
            continue

        df_model = prepare_similarity_data(train_df, k)

        if df_model['similarity'].std() < 1e-9:
            results.append({
                'hold_out_fold': hold_out_fold,
                'k': k,
                'z_value': -999,
                'bic': np.nan,
                'status': 'low_variance',
            })
            continue

        result = fit_glmm(df_model, formula=formula)

        results.append({
            'hold_out_fold': hold_out_fold,
            'k': k,
            'z_value': result['z_value'],
            'bic': result['bic'],
            'status': result['status'],
        })

    return pd.DataFrame(results)


# =============================================================================
# Optuna Integration
# =============================================================================

def optuna_glmm_objective(
    trial,
    df_long_precomputed: pd.DataFrame,
    k_min: float = None,
    k_max: float = None,
    formula: str = None,
    variance_threshold: float = 0.06,
) -> float:
    """Optuna objective function for optimizing k.

    Args:
        trial: Optuna trial object.
        df_long_precomputed: Precomputed distance DataFrame.
        k_min, k_max: k search range.
        formula: GLMM formula.
        variance_threshold: Skip if similarity std is below this.

    Returns:
        z_value if valid, -999.0 otherwise (to penalize failures).
    """
    if k_min is None:
        k_min = config.K_SEARCH_MIN
    if k_max is None:
        k_max = config.K_SEARCH_MAX

    k = trial.suggest_float("k", k_min, k_max, step=config.K_SEARCH_STEP)
    df_model = prepare_similarity_data(df_long_precomputed, k)

    sim_std = df_model['similarity'].std()
    sim_mean = df_model['similarity'].mean()

    if sim_std < variance_threshold:
        trial.set_user_attr("warnings", f"Low variance (std={sim_std:.4f})")
        return -999.0
    if sim_mean < 0.02 or sim_mean > 0.98:
        trial.set_user_attr("warnings", f"Saturation (mean={sim_mean:.4f})")
        return -999.0

    result = fit_glmm(df_model, formula=formula, maxfun=100000)

    warning_str = result['warnings']
    if warning_str:
        trial.set_user_attr("warnings", warning_str)

    if result['status'] == "error":
        trial.set_user_attr("error_msg", result['error_msg'])
        return -999.0

    z = result['z_value']
    trial.set_user_attr("BIC", result['bic'])

    if np.isnan(z) or np.isinf(z) or abs(z) > 15.0:
        trial.set_user_attr("note", f"Unstable Z (val={z})")
        return -999.0

    return z

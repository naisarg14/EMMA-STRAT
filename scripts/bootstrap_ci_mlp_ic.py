######################################################################
## This repository holds the codes for the research paper:          ##
##                                                                  ##
## Paper Title:                                                     ##
##   EMMA-STRAT: A Multi-Omics Based Machine Learning Framework for ##
##   Stratification of Endometrial Carcinoma Molecular Subtypes     ##
##   and MSI Status                                                 ##
##                                                                  ##
## Author:                                                          ##
##   Naisarg Patel                                                  ##
##                                                                  ##
## License: GNU General Public License v3.0 (GPL-3.0)              ##
## Contact: naisargbpatel14<at>gmail<dot>com                        ##
######################################################################

# Description:
#   Computes 95% bootstrap confidence intervals for balanced accuracy,
#   AUC, and macro F1 of the best MLP IC model (RFELASSO, 50 features)
#   across the internal validation set and two external test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.utils import resample
from model_helper import get_sets

# Set random seed for reproducibility
np.random.seed(19)
tf.random.set_seed(19)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "results2601/mlp/RFELASSO_ic_50"
N_BOOTSTRAP = 10000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
os.makedirs("bootstrap_ic", exist_ok=True)

# Model parameters
NAME = "ic"
FS_METHOD = "RFELASSO"
NUM_FS = "50"

print("="*70)
print("Bootstrap 95% Confidence Intervals for MLP IC Model")
print("Feature Selection: Lasso RFE with 50 features (Multiomics)")
print(f"Bootstrap samples: {N_BOOTSTRAP}")
print("="*70)

# ── Load model and data ───────────────────────────────────────────────────────
print("\nLoading model and preprocessing objects...")
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "mlp_model.keras"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
le = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

print(f"Model loaded successfully")
print(f"Classes: {le.classes_}")

# Load datasets
print("\nLoading datasets...")
X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2, X_test3_scaled, y_test3, _ = get_sets(
    NAME, FS_METHOD, NUM_FS, results_folder=None
)

print(f"Validation set: {X_val_scaled.shape[0]} samples, {X_val_scaled.shape[1]} features")
print(f"Test set 2: {X_test2_scaled.shape[0]} samples")
print(f"Test set 3: {X_test3_scaled.shape[0]} samples")


# ── Functions ─────────────────────────────────────────────────────────────────
def compute_metrics(model, X, y_true, le):
    """
    Compute balanced accuracy, AUC, and macro F1 score for given data.

    Returns:
        tuple: (balanced_accuracy, auc, f1_macro)
    """
    # Convert to numpy if pandas DataFrame
    if hasattr(X, 'values'):
        X_np = X.values.astype(np.float32)
    else:
        X_np = X.astype(np.float32)

    # Get predictions (logits from the model)
    logits = model(X_np, training=False).numpy()
    probs = tf.nn.softmax(logits).numpy()
    preds = np.argmax(logits, axis=1)

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, preds)

    # Macro F1 - use labels parameter to only consider classes present in y_true
    unique_classes = np.unique(y_true)
    f1_macro = f1_score(y_true, preds, labels=unique_classes, average='macro')

    # AUC - handle cases where not all classes are present in y_true
    num_classes = len(le.classes_)
    unique_classes_in_sample = np.unique(y_true)

    if num_classes == 2:
        # Binary classification
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        # Multi-class classification (OvR macro-average)
        # Check if all classes are present
        if len(unique_classes_in_sample) < num_classes:
            # Not all classes present - compute AUC only for present classes
            # y_true contains numeric class labels (0, 1, 2, etc.)
            # probs columns correspond to these numeric indices
            # So we can directly use the unique classes as indices
            class_indices = sorted(unique_classes_in_sample)
            probs_filtered = probs[:, class_indices]

            # Renormalize probabilities
            probs_filtered = probs_filtered / probs_filtered.sum(axis=1, keepdims=True)

            # Compute AUC with filtered probabilities
            auc = roc_auc_score(y_true, probs_filtered, labels=class_indices,
                               multi_class='ovr', average='macro')
        else:
            # All classes present
            auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')

    return bal_acc, auc, f1_macro


def bootstrap_ci(model, X, y, le, n_bootstrap=10000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for model metrics.

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        le: Label encoder
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)

    Returns:
        dict: Dictionary containing means, CIs, and bootstrap distributions
    """
    n_samples = len(y)
    alpha = 1 - confidence_level

    # Storage for bootstrap statistics
    bal_acc_scores = []
    auc_scores = []
    f1_scores = []

    print(f"  Running {n_bootstrap} bootstrap iterations...")

    for i in range(n_bootstrap):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i + 1}/{n_bootstrap}")

        # Resample with replacement
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples, random_state=i)
        X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]

        # Compute metrics on bootstrap sample
        bal_acc, auc, f1_macro = compute_metrics(model, X_boot, y_boot, le)

        bal_acc_scores.append(bal_acc)
        auc_scores.append(auc)
        f1_scores.append(f1_macro)

    # Convert to numpy arrays
    bal_acc_scores = np.array(bal_acc_scores)
    auc_scores = np.array(auc_scores)
    f1_scores = np.array(f1_scores)

    # Compute confidence intervals using percentile method
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {
        'balanced_accuracy': {
            'mean': np.mean(bal_acc_scores),
            'std': np.std(bal_acc_scores),
            'ci_lower': np.percentile(bal_acc_scores, lower_percentile),
            'ci_upper': np.percentile(bal_acc_scores, upper_percentile),
            'distribution': bal_acc_scores
        },
        'auc': {
            'mean': np.mean(auc_scores),
            'std': np.std(auc_scores),
            'ci_lower': np.percentile(auc_scores, lower_percentile),
            'ci_upper': np.percentile(auc_scores, upper_percentile),
            'distribution': auc_scores
        },
        'f1_macro': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores),
            'ci_lower': np.percentile(f1_scores, lower_percentile),
            'ci_upper': np.percentile(f1_scores, upper_percentile),
            'distribution': f1_scores
        }
    }

    return results


def print_results(set_name, results, original_metrics):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"{set_name}")
    print(f"{'='*70}")

    print(f"\nOriginal Metrics (on full dataset):")
    print(f"  Balanced Accuracy: {original_metrics[0]:.4f}")
    print(f"  AUC:               {original_metrics[1]:.4f}")
    print(f"  F1 Score (Macro):  {original_metrics[2]:.4f}")

    print(f"\n95% Confidence Intervals (Bootstrap):")
    print(f"\nBalanced Accuracy:")
    print(f"  Mean: {results['balanced_accuracy']['mean']:.4f} ± {results['balanced_accuracy']['std']:.4f}")
    print(f"  95% CI: [{results['balanced_accuracy']['ci_lower']:.4f}, {results['balanced_accuracy']['ci_upper']:.4f}]")

    print(f"\nAUC:")
    print(f"  Mean: {results['auc']['mean']:.4f} ± {results['auc']['std']:.4f}")
    print(f"  95% CI: [{results['auc']['ci_lower']:.4f}, {results['auc']['ci_upper']:.4f}]")

    print(f"\nF1 Score (Macro):")
    print(f"  Mean: {results['f1_macro']['mean']:.4f} ± {results['f1_macro']['std']:.4f}")
    print(f"  95% CI: [{results['f1_macro']['ci_lower']:.4f}, {results['f1_macro']['ci_upper']:.4f}]")


def save_results(output_path, all_results):
    """Save results to file."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Bootstrap 95% Confidence Intervals for MLP IC Model\n")
        f.write("Feature Selection: Lasso RFE with 50 features (Multiomics)\n")
        f.write(f"Bootstrap samples: {N_BOOTSTRAP}\n")
        f.write("="*70 + "\n\n")

        for set_name, data in all_results.items():
            results = data['bootstrap_results']
            original = data['original_metrics']

            f.write(f"\n{'='*70}\n")
            f.write(f"{set_name}\n")
            f.write(f"{'='*70}\n\n")

            f.write(f"Original Metrics (on full dataset):\n")
            f.write(f"  Balanced Accuracy: {original[0]:.4f}\n")
            f.write(f"  AUC:               {original[1]:.4f}\n")
            f.write(f"  F1 Score (Macro):  {original[2]:.4f}\n\n")

            f.write(f"95% Confidence Intervals (Bootstrap):\n\n")

            f.write(f"Balanced Accuracy:\n")
            f.write(f"  Mean: {results['balanced_accuracy']['mean']:.4f} ± {results['balanced_accuracy']['std']:.4f}\n")
            f.write(f"  95% CI: [{results['balanced_accuracy']['ci_lower']:.4f}, {results['balanced_accuracy']['ci_upper']:.4f}]\n\n")

            f.write(f"AUC:\n")
            f.write(f"  Mean: {results['auc']['mean']:.4f} ± {results['auc']['std']:.4f}\n")
            f.write(f"  95% CI: [{results['auc']['ci_lower']:.4f}, {results['auc']['ci_upper']:.4f}]\n\n")

            f.write(f"F1 Score (Macro):\n")
            f.write(f"  Mean: {results['f1_macro']['mean']:.4f} ± {results['f1_macro']['std']:.4f}\n")
            f.write(f"  95% CI: [{results['f1_macro']['ci_lower']:.4f}, {results['f1_macro']['ci_upper']:.4f}]\n\n")

    print(f"\nResults saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
# Store all results
all_results = {}

# ===== INTERNAL VALIDATION SET =====
print("\n" + "="*70)
print("INTERNAL VALIDATION SET")
print("="*70)

# Compute original metrics
original_metrics_val = compute_metrics(model, X_val_scaled, y_val, le)
print(f"\nOriginal metrics computed: Balanced Acc={original_metrics_val[0]:.4f}, "
      f"AUC={original_metrics_val[1]:.4f}, F1={original_metrics_val[2]:.4f}")

# Bootstrap CI
results_val = bootstrap_ci(model, X_val_scaled, y_val, le, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL)
print_results("INTERNAL VALIDATION SET", results_val, original_metrics_val)

all_results['Internal Validation Set'] = {
    'original_metrics': original_metrics_val,
    'bootstrap_results': results_val
}

# ===== EXTERNAL TEST SET 2 =====
print("\n" + "="*70)
print("EXTERNAL TEST SET 2")
print("="*70)

original_metrics_test2 = compute_metrics(model, X_test2_scaled, y_test2, le)
print(f"\nOriginal metrics computed: Balanced Acc={original_metrics_test2[0]:.4f}, "
      f"AUC={original_metrics_test2[1]:.4f}, F1={original_metrics_test2[2]:.4f}")

results_test2 = bootstrap_ci(model, X_test2_scaled, y_test2, le, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL)
print_results("EXTERNAL TEST SET 2", results_test2, original_metrics_test2)

all_results['External Test Set 2'] = {
    'original_metrics': original_metrics_test2,
    'bootstrap_results': results_test2
}

# ===== EXTERNAL TEST SET 3 =====
print("\n" + "="*70)
print("EXTERNAL TEST SET 3")
print("="*70)

original_metrics_test3 = compute_metrics(model, X_test3_scaled, y_test3, le)
print(f"\nOriginal metrics computed: Balanced Acc={original_metrics_test3[0]:.4f}, "
      f"AUC={original_metrics_test3[1]:.4f}, F1={original_metrics_test3[2]:.4f}")

results_test3 = bootstrap_ci(model, X_test3_scaled, y_test3, le, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL)
print_results("EXTERNAL TEST SET 3", results_test3, original_metrics_test3)

all_results['External Test Set 3'] = {
    'original_metrics': original_metrics_test3,
    'bootstrap_results': results_test3
}

# ===== SAVE RESULTS =====
output_file = os.path.join("bootstrap_ic", "bootstrap_confidence_intervals.txt")
save_results(output_file, all_results)

# Save bootstrap distributions as CSV for further analysis
print("\nSaving bootstrap distributions...")
for set_name, data in all_results.items():
    results = data['bootstrap_results']
    df_boot = pd.DataFrame({
        'balanced_accuracy': results['balanced_accuracy']['distribution'],
        'auc': results['auc']['distribution'],
        'f1_macro': results['f1_macro']['distribution']
    })

    csv_name = set_name.lower().replace(' ', '_')
    csv_path = os.path.join("bootstrap_ic", f"bootstrap_distributions_{csv_name}.csv")
    df_boot.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
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
#   Performs feature selection on methylation, miRNA, and mRNA omics data
#   using L1-regularized logistic regression (LASSO). The top-k features are
#   identified by absolute coefficient magnitude and saved to disk.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 19

label = sys.argv[1]
fs_method = "RFELASSO"

np.random.seed(SEED)

folder = f"feature_selection/{fs_method}_{label}"
os.makedirs(folder, exist_ok=True)

target_features = [10] ##[20, 50, 100, 150, 200]

if "ic" in label:
    label_column = "Genomic_Subtype"
elif "msi" in label:
    label_column = "MSI_Status"
else:
    label_column = "Tissue_Type"

drop_columns = [x for x in ["Genomic_Subtype", "MSI_Status", "Tissue_Type"] if x != label_column]

# Configuration for each data type
DATA_CONFIGS = {
    "methyl": {
        "csv_path": "data_extraction/data/set1_methylation_model.csv",
        "feature_prefix": "methyl",
        "target_features": target_features,
    },
    "mirna": {
        "csv_path": "data_extraction/data/set1_mirna_model.csv",
        "feature_prefix": "mirnas",
        "target_features": target_features,
    },
    "mrna": {
        "csv_path": "data_extraction/data/set1_rna_model.csv",
        "feature_prefix": "rna",
        "target_features": target_features,
    },
}

# ── Functions ─────────────────────────────────────────────────────────────────
def _topk_from_lasso_coefs(coef: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k features by absolute coefficient magnitude.

    coef shape is (n_features,) for binary or (n_classes, n_features) for multiclass.
    """
    coef = np.asarray(coef)
    if coef.ndim == 1:
        scores = np.abs(coef)
    else:
        scores = np.max(np.abs(coef), axis=0)

    if k >= scores.shape[0]:
        return np.arange(scores.shape[0])

    # argpartition is O(n) and avoids full sort
    idx = np.argpartition(scores, -k)[-k:]
    # make deterministic ordering by score desc then feature index
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return idx


def run_rfelasso_selection(data_type: str) -> None:
    """Run L1-regularized logistic regression (LASSO) feature selection."""

    config = DATA_CONFIGS[data_type]
    csv_path = config["csv_path"]
    feature_prefix = config["feature_prefix"]
    target_features_list = config["target_features"]

    print(f"\n{'='*70}")
    print(f"Processing {data_type.upper()} data")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path, index_col=0)
    df.dropna(subset=[label_column], inplace=True)
    df.fillna(0, inplace=True)
    print(f"Loaded data shape: {df.shape}")

    for target_k in target_features_list:
        print(f"Processing TARGET_FEATURES = {target_k}")

        y = df[label_column].values
        X = df.drop(columns=[label_column] + drop_columns)

        le = LabelEncoder()
        y = le.fit_transform(y)

        print("\nClass distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Class {le.inverse_transform([cls])[0]}: {cnt} ({cnt/len(y)*100:.1f}%)")

        feature_names = X.columns
        X_v = X.values.astype(np.float32)
        print(f"Features: {X_v.shape[1]}")

        # Train/val split BEFORE scaling to prevent data leakage
        X_train, X_val, y_train, y_val = train_test_split(
            X_v,
            y,
            test_size=0.3,
            random_state=SEED,
            stratify=y,
        )
        print(f"\nTrain: {len(y_train)} samples, Val: {len(y_val)} samples")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        print(f"\nFitting L1 LogisticRegression to select {target_k} features...")

        # L1 logistic regression works for binary and multiclass with saga
        base_model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
            tol=1e-3,
            warm_start=True,
        )

        base_model.fit(X_train, y_train)

        top_idx = _topk_from_lasso_coefs(base_model.coef_, target_k)
        selected_features = feature_names[top_idx]

        print(f"\nSelected {len(selected_features)} {feature_prefix}")

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        )

        clf.fit(X_train[:, top_idx], y_train)
        y_val_pred = clf.predict(X_val[:, top_idx])

        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        val_bacc = balanced_accuracy_score(y_val, y_val_pred)
        print(f"Validation F1: {val_f1:.3f}, Balanced Accuracy: {val_bacc:.3f}")

        print(classification_report(y_val, y_val_pred, target_names=le.classes_))

        confusion = confusion_matrix(y_val, y_val_pred)
        print(f"Confusion Matrix:\n{confusion}")

        output_file = f"{folder}/selected_{feature_prefix}_{fs_method}_{target_k}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

        print(f"Saved to {output_file}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for dt in ["methyl", "mirna", "mrna"]:
        run_rfelasso_selection(dt)

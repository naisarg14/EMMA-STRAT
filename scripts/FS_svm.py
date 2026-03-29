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
#   using Recursive Feature Elimination (RFE) backed by a linear SVM
#   (LinearSVC). Saves the top-k selected feature names to disk.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import LinearSVC

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 19

label = sys.argv[1]
fs_method = "RFESVM"

np.random.seed(SEED)

folder = f"feature_selection2/{fs_method}_{label}"
os.makedirs(folder, exist_ok=True)

target_features = [20] #[20, 50, 100, 150, 200]

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
def run_rfesvm_selection(data_type: str) -> None:
    """Run RFE with a linear SVM for the specified data type."""

    config = DATA_CONFIGS[data_type]
    csv_path = config["csv_path"]
    feature_prefix = config["feature_prefix"]
    target_features_list = config["target_features"]

    print(f"\n{'='*70}")
    print(f"Processing {data_type.upper()} data")
    print(f"{'='*70}")

    # Load data
    df = pd.read_csv(csv_path, index_col=0)
    df.dropna(subset=[label_column], inplace=True)
    df.fillna(0, inplace=True)
    print(f"Loaded data shape: {df.shape}")

    for target_k in target_features_list:
        print(f"Processing TARGET_FEATURES = {target_k}")

        # Prepare features and labels
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

        # Fit scaler on training data only, transform both
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # RFE with linear SVM
        print(f"\nRunning RFE with linear SVM to select {target_k} features...")

        # LinearSVC supports coef_ required by RFE. Use class_weight balanced for imbalance.
        estimator = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=20000,
            dual=True,
        )

        rfe = RFE(
            estimator=estimator,
            n_features_to_select=target_k,
            step=0.1,  # Remove 10% of features at each iteration
            verbose=0,
        )

        rfe.fit(X_train, y_train)

        selected_mask = rfe.support_
        selected_features = feature_names[selected_mask]

        print(f"\nSelected {len(selected_features)} {feature_prefix}")

        # Evaluate on validation set
        y_val_pred = rfe.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        val_bacc = balanced_accuracy_score(y_val, y_val_pred)
        print(f"Validation F1: {val_f1:.3f}, Balanced Accuracy: {val_bacc:.3f}")

        print(classification_report(y_val, y_val_pred, target_names=le.classes_))

        confusion = confusion_matrix(y_val, y_val_pred)
        print(f"Confusion Matrix:\n{confusion}")

        # Save selected features
        output_file = f"{folder}/selected_{feature_prefix}_{fs_method}_{target_k}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

        print(f"Saved to {output_file}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for dt in ["methyl", "mirna", "mrna"]:
        run_rfesvm_selection(dt)

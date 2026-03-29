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
#   using the ANOVA F-test (SelectKBest). Saves the top-k selected feature
#   names and per-feature F-scores/p-values to disk for downstream modelling.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 19

label = sys.argv[1]
fs_method = "ANOVA"

np.random.seed(SEED)

folder = f"feature_selection/{fs_method}_{label}"
os.makedirs(folder, exist_ok=True)

target_features = [10] #[20, 50, 100, 150, 200]

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
def run_anova_selection(data_type: str) -> None:
    """Run ANOVA F-test feature selection for the specified data type."""

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

    for k in target_features_list:
        print(f"Processing TARGET_FEATURES = {k}")

        # Prepare features and labels
        y_raw = df[label_column].values
        X = df.drop(columns=[label_column] + drop_columns)

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        print("\nClass distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Class {le.inverse_transform([cls])[0]}: {cnt} ({cnt/len(y)*100:.1f}%)")

        feature_names = X.columns
        X_v = X.values.astype(np.float32)
        print(f"Features: {X_v.shape[1]}")

        # Train/val split BEFORE scaling to prevent data leakage
        X_train, X_val, y_train, y_val = train_test_split(
            X_v, y, test_size=0.3, random_state=SEED, stratify=y
        )
        print(f"\nTrain: {len(y_train)} samples, Val: {len(y_val)} samples")

        # Fit scaler on training data only, transform both
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Select top-k features by ANOVA F-test (fit on training only)
        k_eff = int(min(k, X_train.shape[1]))
        if k_eff != k:
            print(f"Requested k={k}, but only {X_train.shape[1]} features available; using k={k_eff}")

        print(f"\nRunning ANOVA F-test to select {k_eff} features...")
        selector = SelectKBest(score_func=f_classif, k=k_eff)
        selector.fit(X_train, y_train)

        support_mask = selector.get_support()
        selected_features = feature_names[support_mask]
        print(f"Selected {len(selected_features)} {feature_prefix}")

        # Optional quick model eval (helps sanity-check selection)
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )

        X_train_sel = selector.transform(X_train)
        X_val_sel = selector.transform(X_val)

        clf.fit(X_train_sel, y_train)
        y_val_pred = clf.predict(X_val_sel)

        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        val_bacc = balanced_accuracy_score(y_val, y_val_pred)
        print(f"Validation F1: {val_f1:.3f}, Balanced Accuracy: {val_bacc:.3f}")
        print(classification_report(y_val, y_val_pred, target_names=le.classes_))
        confusion = confusion_matrix(y_val, y_val_pred)
        print(f"Confusion Matrix:\n{confusion}")

        # Save selected features
        output_file = f"{folder}/selected_{feature_prefix}_ANOVA_{k_eff}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        print(f"Saved to {output_file}")

        # Save scores (useful for debugging / downstream analysis)
        scores = selector.scores_
        pvals = selector.pvalues_
        scores_df = pd.DataFrame(
            {
                "feature": feature_names,
                "f_score": scores,
                "p_value": pvals,
                "selected": support_mask,
            }
        )
        scores_df = scores_df.sort_values(["selected", "f_score"], ascending=[False, False])
        scores_file = f"{folder}/scores_{feature_prefix}_ANOVA_{k_eff}.csv"
        scores_df.to_csv(scores_file, index=False)
        print(f"Saved scores to {scores_file}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for data_type in ["methyl", "mirna", "mrna"]:
        run_anova_selection(data_type)



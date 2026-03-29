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
#   Shared utility module used by all model training scripts. Provides
#   get_sets() for loading and splitting multi-omics data, and evaluate_set()
#   / evaluate_lgbm_set() for computing and saving classification metrics.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)

# ── Functions ─────────────────────────────────────────────────────────────────
def get_sets(name, fs_method, num_fs, results_folder=None):

    if "ic" in name:
        label_column = "Genomic_Subtype"
    elif "msi" in name:
        label_column = "MSI_Status"
    else:
        label_column = "Tissue_Type"


    #### Feature selection
    selected_mirnas = [line.strip() for line in open(f"feature_selection/{fs_method}_{name}/selected_mirnas_{fs_method}_{num_fs}.txt")]
    selected_mrna = [line.strip() for line in open(f"feature_selection/{fs_method}_{name}/selected_rna_{fs_method}_{num_fs}.txt")]
    selected_meth = [line.strip() for line in open(f"feature_selection/{fs_method}_{name}/selected_methyl_{fs_method}_{num_fs}.txt")]


    mirna_L = pd.read_csv(f"data_extraction/data/set1_mirna_model.csv", index_col=0)[selected_mirnas + [label_column]]
    mrna_L  = pd.read_csv(f"data_extraction/data/set1_rna_model.csv", index_col=0)[selected_mrna + [label_column]]
    meth_L  = pd.read_csv(f"data_extraction/data/set1_methylation_model.csv", index_col=0)[selected_meth + [label_column]]

    # keep only labeled rows
    mirna_L = mirna_L.dropna(subset=[label_column])
    mrna_L  = mrna_L.dropna(subset=[label_column])
    meth_L  = meth_L.dropna(subset=[label_column])

    # Combine modalities by index (inner join) and validate labels
    combined_df = pd.concat([mirna_L, mrna_L, meth_L], axis=1, join='inner')


    X_combined = combined_df.drop(columns=[label_column])
    y = combined_df[label_column]
    y = y.apply(
        lambda x: x.iloc[0] if x.nunique() == 1 else np.nan,
        axis=1
    )
    print(f"Labeled samples: {combined_df.shape}")
    print(f"X shape: {X_combined.shape}")
    print(f"y shape: {y.shape}")



    # Encode labels and do a train/val split on the combined DataFrame (preserves index)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train_df, X_val_df, y_train_enc, y_val_enc = train_test_split(
        X_combined, y_enc, test_size=0.30, stratify=y_enc, random_state=19
    )


    # Single scaler for all features fit on training portion
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_df),
        index=X_train_df.index,
        columns=X_train_df.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_df),
        index=X_val_df.index,
        columns=X_val_df.columns
    )

    if results_folder is not None:
        #save scaler
        scaler_path = os.path.join(results_folder, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        label_encoder_path = os.path.join(results_folder, "label_encoder.pkl")
        joblib.dump(le, label_encoder_path)

    if label_column == "Tissue_Type":
        return (X_tr_scaled, y_train_enc, X_val_scaled, y_val_enc,
                None, None,
                None, None, le)

    mirna_test2 = pd.read_csv(f"data_extraction/data/set2_mirna_model.csv", index_col=0)[selected_mirnas + [label_column]]
    mrna_test2  = pd.read_csv(f"data_extraction/data/set2_rna_model.csv", index_col=0)[selected_mrna + [label_column]]
    meth_test2  = pd.read_csv(f"data_extraction/data/set2_methylation_model.csv", index_col=0)[selected_meth + [label_column]]

    mirna_test2 = mirna_test2.dropna(subset=[label_column])
    mrna_test2 = mrna_test2.dropna(subset=[label_column])
    meth_test2 = meth_test2.dropna(subset=[label_column])
    # Combine test modalities by index and validate labels
    combined_df_test = pd.concat([mirna_test2, mrna_test2, meth_test2], axis=1, join='inner')


    X_test2_combined = combined_df_test.drop(columns=[label_column])
    y_test2 = combined_df_test[label_column]

    y_test2 = y_test2.apply(
        lambda x: x.iloc[0] if x.nunique() == 1 else np.nan,
        axis=1
    )

    # Use combined test features (already concatenated) and align columns
    test2_combined_df = X_test2_combined.reindex(columns=X_tr_scaled.columns)
    X_test2_scaled = pd.DataFrame(
        scaler.transform(test2_combined_df),
        columns=test2_combined_df.columns,
        index=test2_combined_df.index
    )

    y_test2_enc = le.transform(y_test2)

    mirna_test3 = pd.read_csv(f"data_extraction/data/set3_mirna_model.csv", index_col=0)[selected_mirnas + [label_column]]
    mrna_test3  = pd.read_csv(f"data_extraction/data/set3_rna_model.csv", index_col=0)[selected_mrna + [label_column]]
    meth_test3  = pd.read_csv(f"data_extraction/data/set3_methylation_model.csv", index_col=0)[selected_meth + [label_column]]

    mirna_test3 = mirna_test3.dropna(subset=[label_column])
    mrna_test3 = mrna_test3.dropna(subset=[label_column])
    meth_test3 = meth_test3.dropna(subset=[label_column])
    # Combine test modalities by index and validate labels
    combined_df_test = pd.concat([mirna_test3, mrna_test3, meth_test3], axis=1, join='inner')


    X_test3_combined = combined_df_test.drop(columns=[label_column])
    y_test3 = combined_df_test[label_column]
    y_test3 = y_test3.apply(
        lambda x: x.iloc[0] if x.nunique() == 1 else np.nan,
        axis=1
    )

    # Use combined test features (already concatenated) and align columns
    test3_combined_df = X_test3_combined.reindex(columns=X_tr_scaled.columns)
    X_test3_scaled = pd.DataFrame(
        scaler.transform(test3_combined_df),
        columns=test3_combined_df.columns,
        index=test3_combined_df.index
    )

    y_test3_enc = le.transform(y_test3)

    y_train_enc = pd.Series(y_train_enc, index=X_train_df.index, name='label')
    y_val_enc = pd.Series(y_val_enc, index=X_val_df.index, name='label')

    y_test2_enc = pd.Series(y_test2_enc, index=test2_combined_df.index, name='label')
    y_test3_enc = pd.Series(y_test3_enc, index=test3_combined_df.index, name='label')


    return (X_tr_scaled, y_train_enc, X_val_scaled, y_val_enc,
            X_test2_scaled, y_test2_enc,
            X_test3_scaled, y_test3_enc, le)


def evaluate_set(model, X, y_true, set_name, le, save_file=None):
    # Get predictions (logits)
    import tensorflow as tf
    logits = model.predict(X, verbose=0)

    # Convert logits to probabilities using softmax
    probs = tf.nn.softmax(logits).numpy()

    # Get class predictions
    preds = np.argmax(probs, axis=1)

    num_classes = len(le.classes_)

    print("\n" + "="*50)
    print(f"{set_name} EVALUATION")
    print("="*50)

    report = classification_report(y_true, preds, target_names=le.classes_)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")

    print(report)
    print("Balanced accuracy:", bal_acc)
    print("Macro F1:", macro_f1)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, preds, labels=np.arange(num_classes))
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{cls}" for cls in le.classes_],
        columns=[f"Pred_{cls}" for cls in le.classes_]
    )
    print(cm_df)

    auc_results = []
    if num_classes == 2:
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])

        auc = roc_auc_score(y_true, probs[:, 1])
        line = f"AUC: {auc:.4f} ({le.classes_[1]} vs {le.classes_[0]})"
        print(line)
        auc_results = [line]
    else:
        print("\nAUC per class (One-vs-Rest):")
        auc_results = []

        for i, class_name in enumerate(le.classes_):
            y_binary = (y_true == i).astype(int)
            auc = roc_auc_score(y_binary, probs[:, i])
            line = f"  Class {class_name}: {auc:.4f}"
            print(line)
            auc_results.append(line)

        # Macro-average AUC
        macro_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        macro_line = f"  Macro-average AUC: {macro_auc:.4f}"
        print(macro_line)
        auc_results.append(macro_line)

    # Save results if filename provided
    if save_file:
        with open(save_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write(f"{set_name} EVALUATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(report + "\n")
            f.write(f"Balanced accuracy: {bal_acc}\n")
            f.write(f"Macro F1: {macro_f1}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(cm_df.to_string() + "\n\n")
            f.write("AUC per class:\n")
            for auc_line in auc_results:
                f.write(auc_line + "\n")

        # Save confusion matrix as CSV
        cm_df.to_csv(save_file.replace(".txt", "_confusion_matrix.csv"))


def evaluate_lgbm_set(model, X, y_true, set_name, le, save_file=None):
    preds = model.predict(X)
    num_classes = len(le.classes_)

    print("\n" + "="*50)
    print(f"{set_name} EVALUATION")
    print("="*50)

    report = classification_report(y_true, preds, target_names=le.classes_)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")

    print(report)
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, preds, labels=np.arange(num_classes))
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{cls}" for cls in le.classes_],
        columns=[f"Pred_{cls}" for cls in le.classes_]
    )
    print(cm_df)

    # AUC calculation
    probs = model.predict_proba(X)

    if num_classes == 2:
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])

        auc = roc_auc_score(y_true, probs[:, 1])
        line = f"AUC: {auc:.4f} ({le.classes_[1]} vs {le.classes_[0]})"
        print(line)
        auc_results = [line]

    else:
        print("\nAUC per class (One-vs-Rest):")
        auc_results = []

        for i, class_name in enumerate(le.classes_):
            y_binary = (y_true == i).astype(int)
            auc = roc_auc_score(y_binary, probs[:, i])
            line = f"  Class {class_name}: {auc:.4f}"
            print(line)
            auc_results.append(line)

        # Macro-average AUC
        macro_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        macro_line = f"  Macro-average AUC: {macro_auc:.4f}"
        print(macro_line)
        auc_results.append(macro_line)

    # Save results if filename provided
    if save_file:
        with open(save_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write(f"{set_name} EVALUATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(report + "\n")
            f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(cm_df.to_string() + "\n\n")
            f.write("AUC:\n")
            for auc_line in auc_results:
                f.write(auc_line + "\n")

        # Save confusion matrix as CSV
        cm_df.to_csv(save_file.replace(".txt", "_confusion_matrix.csv"))

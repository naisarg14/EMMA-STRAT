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
# Trains single-omics LightGBM classifiers with Optuna hyperparameter tuning.
# Iterates over miRNA, RNA, and methylation feature subsets independently,
# optimizing each via 5-fold CV and evaluating on validation and external test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import numpy as np
import pandas as pd
import os
from model_helper import evaluate_lgbm_set, get_sets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
import joblib
import lightgbm as lgb

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)


name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]

if "ic" in name:
    label_column = "Genomic_Subtype"
elif "msi" in name:
    label_column = "MSI_Status"
else:
    label_column = "Tissue_Type"

results_folder = f"single/lgbm/{fs_method}_{name}_{num_fs}"
os.makedirs(results_folder, exist_ok=True)

X_tr_scaled_all, y_tr, X_val_scaled_all, y_val, X_test2_scaled_all, y_test2_enc, X_test3_scaled_all, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)

num_classes = len(le.classes_)

print("\nOptimizing LightGBM hyperparameters with Optuna...")

# ── Main ──────────────────────────────────────────────────────────────────────
for selected_fs in ["mirnas", "rna", "methyl"]:
    results_folder = f"single/lgbm/{fs_method}_{name}_{num_fs}/{selected_fs}"
    os.makedirs(results_folder, exist_ok=True)
    selected_fs_list = [line.strip() for line in open(f"feature_selection/{fs_method}_{name}/selected_{selected_fs}_{fs_method}_{num_fs}.txt")]

    X_tr_scaled = X_tr_scaled_all[selected_fs_list]
    X_val_scaled = X_val_scaled_all[selected_fs_list]
    X_test2_scaled = X_test2_scaled_all[selected_fs_list]
    X_test3_scaled = X_test3_scaled_all[selected_fs_list]

    # ── Functions ─────────────────────────────────────────────────────────────
    def create_lgbm_model(num_classes, params):
        """Helper to create LightGBM model with proper config"""
        base_params = {
            "random_state": 19,
            "verbosity": -1,
            "n_jobs": -1,
            "class_weight": "balanced",
        }

        if num_classes == 2:
            return lgb.LGBMClassifier(
                objective="binary",
                **base_params,
                **params,
            )
        else:
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                **base_params,
                **params,
            )


    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),  # Reduced upper bound
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),  # Removed step for finer control
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # Removed step, expanded range
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Removed step, expanded range
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),  # Log scale for regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),  # Log scale for regularization
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),  # Additional regularization
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10) if trial.params["subsample"] < 1.0 else 0,  # Only if subsampling
        }

        model = create_lgbm_model(num_classes, params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
        scores = []

        for train_idx, val_idx in cv.split(X_tr_scaled, y_tr):
            X_fold_train = X_tr_scaled.iloc[train_idx]
            y_fold_train = y_tr.iloc[train_idx]
            X_fold_val = X_tr_scaled.iloc[val_idx]
            y_fold_val = y_tr.iloc[val_idx]

            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
            )


            preds = model.predict(X_fold_val)
            scores.append(f1_score(y_fold_val, preds, average="macro"))

        return float(np.mean(scores))


    study = optuna.create_study(direction="maximize",
                storage=f"sqlite:///optuna_study.db",
                study_name=f"lgbm_single_optimization_{name}_{fs_method}_{num_fs}_{selected_fs}",
                load_if_exists=True
            )

    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )

    # Display results
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest CV Macro-F1: {study.best_value:.4f}")
    print(f"Number of trials completed: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")


    # Save best parameters with additional info
    with open(f"{results_folder}best_params.txt", "w") as f:
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")


    print("\nTraining final LightGBM with optimized parameters...")

    # Clean up conditional parameters
    final_params = best_params.copy()
    if final_params.get("subsample", 1.0) >= 1.0:
        final_params.pop("subsample_freq", None)

    lgbm_model = create_lgbm_model(num_classes, final_params)
    lgbm_model.fit(
        X_tr_scaled,
        y_tr,
        callbacks=[lgb.log_evaluation(period=100)]
    )

    #save model
    joblib.dump(lgbm_model, os.path.join(results_folder, "lgbm_model.pkl"))
    print("Training complete!")



    # Internal validation
    evaluate_lgbm_set(lgbm_model, X_val_scaled, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
                save_file=os.path.join(results_folder, "validation_set_results.txt"))

    if name != "nt":
        evaluate_lgbm_set(lgbm_model, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                    save_file=os.path.join(results_folder, "test_set2_results.txt"))

        evaluate_lgbm_set(lgbm_model, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                    save_file=os.path.join(results_folder, "test_set3_results.txt"))



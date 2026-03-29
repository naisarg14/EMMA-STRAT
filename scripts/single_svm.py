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
# Trains single-omics Support Vector Machine classifiers with Optuna hyperparameter tuning.
# Iterates over miRNA, RNA, and methylation feature subsets independently,
# searching kernel, C, and gamma via 5-fold CV and evaluating on validation and test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import sys
from model_helper import evaluate_lgbm_set, get_sets
import optuna
import os
from sklearn.model_selection import cross_val_score
from optuna.pruners import MedianPruner
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)

name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]


results_folder = f"single/svm/{fs_method}_{name}_{num_fs}/"
os.makedirs(results_folder, exist_ok=True)

X_tr_scaled_all, y_tr, X_val_scaled_all, y_val, X_test2_scaled_all, y_test2_enc, X_test3_scaled_all, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)
print("\nOptimizing SVM hyperparameters with Optuna...")

# ── Main ──────────────────────────────────────────────────────────────────────
for selected_fs in ["mirnas", "rna", "methyl"]:
    results_folder = f"single/svm/{fs_method}_{name}_{num_fs}/{selected_fs}"
    os.makedirs(results_folder, exist_ok=True)
    selected_fs_list = [line.strip() for line in open(f"feature_selection/{fs_method}_{name}/selected_{selected_fs}_{fs_method}_{num_fs}.txt")]

    X_tr_scaled = X_tr_scaled_all[selected_fs_list]
    X_val_scaled = X_val_scaled_all[selected_fs_list]
    X_test2_scaled = X_test2_scaled_all[selected_fs_list]
    X_test3_scaled = X_test3_scaled_all[selected_fs_list]

    # ── Functions ─────────────────────────────────────────────────────────────
    def objective(trial):
        # Hyperparameter search space
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])

        # Expanded gamma options: categorical + numeric
        gamma_type = trial.suggest_categorical("gamma_type", ["scale", "auto", "numeric"])
        if gamma_type == "numeric":
            gamma = trial.suggest_float("gamma_value", 1e-4, 1e1, log=True)
        else:
            gamma = gamma_type

        # Conditional parameters based on kernel
        degree = 3
        coef0 = 0.0

        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
            coef0 = trial.suggest_float("coef0", 0.0, 10.0)
        elif kernel == "sigmoid":
            coef0 = trial.suggest_float("coef0", 0.0, 10.0)

        # Build model
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=True,
            class_weight="balanced",
            random_state=19,
        )

        # Cross-validation with parallel processing
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
        scores = cross_val_score(
            model,
            X_tr_scaled,
            y_tr,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1  # Use all available cores
        )

        return scores.mean()


    # Create study with pruning for faster optimization
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        storage=f"sqlite:///optuna_study_pc.db",
        study_name=f"svm_single_optimization_{name}_{fs_method}_{num_fs}_{selected_fs}",
        load_if_exists=True
    )


    desired_trials = 50

    if len(study.trials) < desired_trials:
        study.optimize(objective, n_trials=(desired_trials-len(study.trials)), show_progress_bar=True)


    # Display results
    best_params = study.best_params
    print("\nBest SVM Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest CV Macro-F1: {study.best_value:.4f}")
    print(f"Number of trials completed: {len(study.trials)}")

    # Save best parameters with additional info
    with open(f"{results_folder}best_params.txt", "w") as f:
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")

    # Build final model with best parameters
    final_kernel = best_params["kernel"]
    final_degree = best_params.get("degree", 3)
    final_coef0 = best_params.get("coef0", 0.0)

    # Handle gamma parameter
    if best_params.get("gamma_type") == "numeric":
        final_gamma = best_params["gamma_value"]
    else:
        final_gamma = best_params["gamma_type"]

    svm_model = SVC(
        C=best_params["C"],
        kernel=final_kernel,
        gamma=final_gamma,
        degree=final_degree,
        coef0=final_coef0,
        probability=True,
        class_weight="balanced",
        random_state=19,
    )

    print("\nFinal model created with best hyperparameters.")
    svm_model.fit(X_tr_scaled, y_tr)
    print("Training complete!")

    joblib.dump(svm_model, os.path.join(results_folder, "svm_model.pkl"))

    evaluate_lgbm_set(svm_model, X_val_scaled, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
                save_file=os.path.join(results_folder, "validation_set_results.txt"))

    if name != "nt":
        evaluate_lgbm_set(svm_model, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                    save_file=os.path.join(results_folder, "test_set2_results.txt"))

        evaluate_lgbm_set(svm_model, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                    save_file=os.path.join(results_folder, "test_set3_results.txt"))

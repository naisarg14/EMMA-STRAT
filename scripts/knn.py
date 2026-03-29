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
# Trains a multi-omics K-Nearest Neighbors classifier with Optuna hyperparameter tuning.
# Searches neighbor count, distance metric, and weighting scheme via 5-fold stratified CV,
# then fits a final model and evaluates on internal validation and external test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import sys
import optuna
import os
from model_helper import evaluate_lgbm_set, get_sets
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)



name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]

results_folder = f"results2601/knn/{fs_method}_{name}_{num_fs}/"
os.makedirs(results_folder, exist_ok=True)


X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)


print("\nOptimizing KNN hyperparameters with Optuna...")

# ── Functions ─────────────────────────────────────────────────────────────────
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])


    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])

    if metric == 'minkowski':
        p = trial.suggest_int('p', 1, 5)
    else:
        p = 2

    leaf_size = trial.suggest_int('leaf_size', 10, 50)

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric=metric,
        leaf_size=leaf_size,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    scores = cross_val_score(
        model,
        X_tr_scaled,
        y_tr,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )

    return scores.mean()

# ── Main ──────────────────────────────────────────────────────────────────────
# Optimize with pruning and more trials
study = optuna.create_study(direction='maximize')

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


with open(f"{results_folder}best_params.txt", "w") as f:
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")
    f.write(f"Number of trials: {len(study.trials)}\n")


print("\nTraining final KNN with optimized parameters...")

# Handle conditional parameters
final_params = best_params.copy()
if best_params['metric'] != 'minkowski':
    final_params.pop('p', None)

knn_model = KNeighborsClassifier(
    **final_params,
    n_jobs=-1
)

knn_model.fit(X_tr_scaled, y_tr)

joblib.dump(knn_model, os.path.join(results_folder, "knn_model.pkl"))

print("Training complete!")

evaluate_lgbm_set(knn_model, X_val_scaled, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
             save_file=os.path.join(results_folder, "validation_set_results.txt"))

if name != "nt":
    evaluate_lgbm_set(knn_model, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                save_file=os.path.join(results_folder, "test_set2_results.txt"))

    evaluate_lgbm_set(knn_model, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                save_file=os.path.join(results_folder, "test_set3_results.txt"))

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
# Trains a multi-omics Random Forest classifier with Optuna hyperparameter tuning.
# Searches tree structure and sampling parameters via 5-fold stratified CV,
# then fits a final model and evaluates on internal validation and external test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import sys
from model_helper import evaluate_lgbm_set, get_sets
import optuna
from sklearn.model_selection import cross_val_score
from optuna.pruners import MedianPruner
import joblib
import os

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)


name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]


results_folder = f"results2601/rf/{fs_method}_{name}_{num_fs}/"
os.makedirs(results_folder, exist_ok=True)

X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)

print("\nOptimizing Random Forest hyperparameters with Optuna...")

# ── Functions ─────────────────────────────────────────────────────────────────
def objective(trial):
    """Optuna objective function for RF optimization."""
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    # Additional useful parameters
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.2)
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if bootstrap else None

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        min_impurity_decrease=min_impurity_decrease,
        class_weight='balanced',
        random_state=19,
        n_jobs=-1,
    )

    # Use cross_val_score for cleaner code
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    scores = cross_val_score(
        model,
        X_tr_scaled,
        y_tr,
        cv=cv,
        scoring='f1_macro',
        n_jobs=1
    )

    return scores.mean()

# ── Main ──────────────────────────────────────────────────────────────────────
# Optimize with pruning (RF can benefit from early stopping)
study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3)
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


# Save best parameters
with open(f"{results_folder}best_params.txt", "w") as f:
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")


print("\nTraining final Random Forest with optimized parameters...")

final_params = best_params.copy()
if not best_params.get('bootstrap', True):
    final_params.pop('max_samples', None)

rf = RandomForestClassifier(
    **final_params,
    class_weight='balanced',
    random_state=19,
    n_jobs=-1,
    verbose=0
)

rf.fit(X_tr_scaled, y_tr)

joblib.dump(rf, os.path.join(results_folder, "rf_model.pkl"))

evaluate_lgbm_set(rf, X_val_scaled, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
             save_file=os.path.join(results_folder, "validation_set_results.txt"))

if name != "nt":
    evaluate_lgbm_set(rf, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                save_file=os.path.join(results_folder, "test_set2_results.txt"))

    evaluate_lgbm_set(rf, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                save_file=os.path.join(results_folder, "test_set3_results.txt"))

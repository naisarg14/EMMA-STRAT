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
# Trains a multi-omics MLP neural network classifier with Optuna hyperparameter tuning.
# Uses stratified 5-fold cross-validation with early stopping on macro-F1,
# then evaluates the final model on internal validation and external test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from model_helper import get_sets, evaluate_set
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys
import optuna
import os
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)
tf.random.set_seed(19)


name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]


results_folder = f"results1302/mlp_smote/{fs_method}_{name}_{num_fs}/"
os.makedirs(results_folder, exist_ok=True)

X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)



# Convert to numpy arrays for TensorFlow
X_tr_np = X_tr_scaled.values.astype(np.float32)
X_val_np = X_val_scaled.values.astype(np.float32)
y_tr = y_tr.values.astype(np.int32)
y_val = y_val.values.astype(np.int32)


redo = False
if os.path.exists(os.path.join(results_folder, "mlp_model.keras")) and not redo:
    mlp_model = tf.keras.models.load_model(os.path.join(results_folder, "mlp_model.keras"))
    evaluate_set(mlp_model, X_val_np, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
                save_file=f"{results_folder}validation_set_results.txt")



    if name != "nt":
        evaluate_set(mlp_model, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                    save_file=f"{results_folder}test_set2_results.txt")

        evaluate_set(mlp_model, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                save_file=f"{results_folder}test_set3_results.txt")

    sys.exit(0)

num_classes = len(le.classes_)

# ── Functions ─────────────────────────────────────────────────────────────────
def build_mlp(input_dim, hidden_dims, num_classes, dropout=0.3, l2_reg=0.0):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for h in hidden_dims:
        model.add(tf.keras.layers.Dense(
            h,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation=None))
    return model


def train_model(model, X_train, y_train, batch_size, X_val, y_val, epochs=100,
                lr=3e-4, patience=15, class_weights=None, trial=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[]
    )

    # Callback to compute validation macro F1
    class ValF1Callback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            X_val_cb, y_val_cb = self.validation_data
            probs = self.model.predict(X_val_cb, verbose=0)
            preds = np.argmax(probs, axis=1)
            f1 = f1_score(y_val_cb, preds, average='macro')
            logs['val_macro_f1'] = f1

    # Build callbacks
    val_f1_cb = ValF1Callback(validation_data=(X_val, y_val))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_macro_f1',
        patience=patience,
        restore_best_weights=True,
        verbose=0,
        mode='max'
    )

    callbacks = [val_f1_cb, early_stop]

    # Add Optuna pruning callback if trial provided
    if trial is not None:
        pruning_callback = TFKerasPruningCallback(trial, 'val_macro_f1')
        callbacks.append(pruning_callback)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    # Return best validation macro F1
    val_f1s = history.history.get('val_macro_f1', [])
    return float(max(val_f1s)) if val_f1s else 0.0


print("\nOptimizing MLP hyperparameters with Optuna...")


def objective(trial):
    # Hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 2, 4)  # Reduced max for faster training
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)  # Removed step for finer control
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)

    hidden_dims = [hidden_dim] * num_layers
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_tr_np, y_tr)):
        X_fold_train = X_tr_np[train_idx]
        y_fold_train = y_tr[train_idx]
        X_fold_val = X_tr_np[val_idx]
        y_fold_val = y_tr[val_idx]

        # Clear session to prevent memory leaks
        tf.keras.backend.clear_session()

        model = build_mlp(
            X_tr_np.shape[1],
            hidden_dims,
            num_classes,
            dropout=dropout,
            l2_reg=l2_reg
        )

        # Compute class weights for fold
        classes = np.unique(y_fold_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_fold_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

        # Train with pruning callback
        try:
            val_f1 = train_model(
                model, X_fold_train, y_fold_train, batch_size,
                X_fold_val, y_fold_val, epochs=50, lr=lr,
                patience=10, class_weights=class_weights,
                trial=trial  # Pass trial for pruning
            )
        except optuna.TrialPruned:
            raise

        scores.append(val_f1)

        # Check if trial should be pruned after this fold
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))

# ── Main ──────────────────────────────────────────────────────────────────────
# Optimize with pruning
study = optuna.create_study(
    storage=f"sqlite:///optuna_study_server.db",
    study_name=f"mlp_optimization_{name}_{fs_method}_{num_fs}",
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1  # Can prune after first CV fold
    )
)

study.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True,
    gc_after_trial=True
)

# Display results
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest CV Macro-F1: {study.best_value:.4f}")
print(f"Number of trials completed: {len(study.trials)}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")


# Save best parameters
with open(f"{results_folder}best_params.txt", "w") as f:
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")
    f.write(f"Number of trials: {len(study.trials)}\n")
    f.write(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")


print("\nTraining final MLP with optimized parameters...")

# Clear any previous models
tf.keras.backend.clear_session()

hidden_dims = [best_params['hidden_dim']] * best_params['num_layers']

mlp_model = build_mlp(
    X_tr_np.shape[1],
    hidden_dims,
    num_classes,
    dropout=best_params['dropout'],
    l2_reg=best_params['l2_reg']
)

# Compute final class weights
classes_final = np.unique(y_tr)
cw_final = compute_class_weight('balanced', classes=classes_final, y=y_tr)
class_weights_final = {int(c): float(w) for c, w in zip(classes_final, cw_final)}


train_model(
    mlp_model, X_tr_np, y_tr,
    best_params['batch_size'],
    X_val_np, y_val,
    epochs=150,
    lr=best_params['lr'],
    patience=20,  # Increased patience for final training
    class_weights=class_weights_final,
    trial=None  # No pruning for final model
)

print("Training complete!")
#save model
model_path = os.path.join(results_folder, "mlp_model.keras")
mlp_model.save(model_path)
print(f"\nSaved final model to {model_path}")

evaluate_set(mlp_model, X_val_np, y_val, "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
             save_file=f"{results_folder}validation_set_results.txt")



if name != "nt":
    evaluate_set(mlp_model, X_test2_scaled, y_test2_enc, f"EXTERNAL TEST SET (Set 2)", le,
                save_file=f"{results_folder}test_set2_results.txt")

    evaluate_set(mlp_model, X_test3_scaled, y_test3_enc, f"EXTERNAL TEST SET (Set 3)", le,
                save_file=f"{results_folder}test_set3_results.txt")

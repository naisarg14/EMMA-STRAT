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
# Trains a multi-omics Graph Neural Network (TF-GNN / mt_albis) classifier with Optuna tuning.
# Builds k-NN sample graphs with Gaussian-kernel edge weights, optimizes architecture
# and training hyperparameters via 5-fold CV, then evaluates on validation and test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
from model_helper import get_sets
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import mt_albis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
import sys
import optuna
from sklearn.utils.class_weight import compute_class_weight
from optuna.integration import TFKerasPruningCallback
from sklearn.neighbors import kneighbors_graph

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(19)
tf.random.set_seed(19)

NODE_SET_NAME = "nodes"
EDGE_SET_NAME = "edges"
FEATURE_NAME = "features"

name = sys.argv[1]
fs_method = sys.argv[2]
num_fs = sys.argv[3]

results_folder = f"results2601/gnn/{fs_method}_{name}_{num_fs}/"
os.makedirs(results_folder, exist_ok=True)

X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs, results_folder)

# Convert to numpy arrays for TensorFlow
X_tr_np = X_tr_scaled.values.astype(np.float32)
X_val_np = X_val_scaled.values.astype(np.float32)
y_tr = y_tr.values.astype(np.int32)
y_val = y_val.values.astype(np.int32)

# ── Functions ─────────────────────────────────────────────────────────────────
def build_graph_tensor_from_knn(X, k_neighbors=5, include_self=True):
    """
    Build a TF-GNN GraphTensor from features using weighted k-NN edges.

    Uses Gaussian kernel to convert distances to similarity weights:
    weight = exp(-(distance^2) / (2*sigma^2))
    where sigma = mean of all k-NN distances
    """
    num_nodes = X.shape[0]

    # Get k-NN distances (not just connectivity)
    knn = kneighbors_graph(
        X,
        k_neighbors,
        mode='distance',  # Get actual distances
        include_self=include_self
    )

    # Convert distances to similarities using Gaussian/RBF kernel
    distances = knn.data
    sigma = np.mean(distances)  # Adaptive bandwidth
    similarities = np.exp(-(distances**2) / (2 * sigma**2))
    knn.data = similarities.astype(np.float32)

    src, tgt = knn.nonzero()
    edge_weights = knn.data

    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            NODE_SET_NAME: tfgnn.NodeSet.from_fields(
                sizes=[num_nodes],
                features={FEATURE_NAME: tf.convert_to_tensor(X)}
            )
        },
        edge_sets={
            EDGE_SET_NAME: tfgnn.EdgeSet.from_fields(
                sizes=[len(src)],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(NODE_SET_NAME, tf.convert_to_tensor(src, dtype=tf.int32)),
                    target=(NODE_SET_NAME, tf.convert_to_tensor(tgt, dtype=tf.int32))
                ),
                features={
                    "weight": tf.convert_to_tensor(edge_weights)
                }
            )
        }
    )



def evaluate_set_gnn(model, X, k_neighbors, y_true, set_name, le, save_file=None):
    """Evaluate TF-GNN model on a graph constructed from X with k-NN edges."""
    graph = build_graph_tensor_from_knn(X, k_neighbors=k_neighbors)
    # Call model directly instead of predict to avoid unbatching scalar GraphTensor
    logits = model(graph, training=False).numpy()

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
        # Binary: use probability of positive class
        auc = roc_auc_score(y_true, probs[:, 1])
        line = f"AUC: {auc:.4f} ({le.classes_[1]} vs {le.classes_[0]})"
        print(line)
        auc_results = [line]
    else:
        # Multiclass: one-vs-rest
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

    return macro_f1


num_classes = len(le.classes_)




def build_gnn(graph_tensor_spec, hidden_dim, num_layers, num_classes, dropout=0.3, l2_reg=0.0):
    """
    Build a GNN using TensorFlow GNN (mt_albis) with weighted edge aggregation.
    Uses edge weights from k-NN similarity for message passing.
    """
    inputs = tf.keras.Input(type_spec=graph_tensor_spec)

    def set_initial_node_states(node_set, *, node_set_name):
        del node_set_name
        x = node_set[FEATURE_NAME]
        x = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return {tfgnn.HIDDEN_STATE: x}

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_states)(inputs)

    for _ in range(num_layers):
        graph = mt_albis.MtAlbisGraphUpdate(
            units=hidden_dim,
            message_dim=max(16, hidden_dim // 2),
            receiver_tag=tfgnn.TARGET,
            node_set_names=[NODE_SET_NAME],
            attention_type="none",
            simple_conv_reduce_type="sum",  # Use sum to respect edge weights
            normalization_type="layer",
            next_state_type="residual",
            state_dropout_rate=dropout,
            l2_regularization=l2_reg,
        )(graph)

    node_states = tfgnn.keras.layers.Readout(
        node_set_name=NODE_SET_NAME,
        feature_name=tfgnn.HIDDEN_STATE
    )(graph)

    logits = tf.keras.layers.Dense(num_classes, activation=None)(node_states)
    return tf.keras.Model(inputs, logits)



def train_model(model, graph_train, y_train, graph_val, y_val,
                epochs=100, lr=3e-4, patience=15, class_weights=None, trial=None):
    """
    Train model on single graph with node labels.
    graph_train and graph_val are scalar GraphTensors.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[]
    )

    graph_train_batched = graph_train.merge_batch_to_components()
    graph_val_batched = graph_val.merge_batch_to_components()

    # Now wrap in dataset
    train_ds = tf.data.Dataset.from_tensors((graph_train_batched, y_train)).repeat()
    val_ds = tf.data.Dataset.from_tensors((graph_val_batched, y_val)).repeat()

    # Callback to compute validation macro F1
    class ValF1Callback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.graph_val, self.y_val = validation_data

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Call model directly instead of predict to avoid unbatching
            logits = self.model(self.graph_val, training=False)
            probs = tf.nn.softmax(logits).numpy()
            preds = np.argmax(probs, axis=1)
            f1 = f1_score(self.y_val, preds, average='macro')
            logs['val_macro_f1'] = f1

    # Build callbacks
    val_f1_cb = ValF1Callback(validation_data=(graph_val_batched, y_val))
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
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=1,  # Single graph, one step per epoch
        validation_steps=1,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    # Return best validation macro F1
    val_f1s = history.history.get('val_macro_f1', [])
    return float(max(val_f1s)) if val_f1s else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
print("\nOptimizing GNN hyperparameters with Optuna...")


def objective(trial):
    # Hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 2, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    k_neighbors = trial.suggest_int('k_neighbors', 5, 20)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_tr_np, y_tr)):
        X_fold_train = X_tr_np[train_idx]
        y_fold_train = y_tr[train_idx]
        X_fold_val = X_tr_np[val_idx]
        y_fold_val = y_tr[val_idx]

        # Build graph tensors from k-NN
        graph_train = build_graph_tensor_from_knn(X_fold_train, k_neighbors=k_neighbors)
        graph_val = build_graph_tensor_from_knn(X_fold_val, k_neighbors=k_neighbors)

        # Clear session to prevent memory leaks
        tf.keras.backend.clear_session()

        model = build_gnn(
            graph_train.spec,
            hidden_dim,
            num_layers,
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
                model, graph_train, y_fold_train,
                graph_val, y_fold_val, epochs=50, lr=lr,
                patience=10, class_weights=class_weights,
                trial=trial
            )
        except optuna.TrialPruned:
            raise

        scores.append(val_f1)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


# Optimize with pruning
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1
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
print("\n" + "="*50)
print("BEST GNN HYPERPARAMETERS")
print("="*50)
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest CV Macro-F1: {study.best_value:.4f}")
print(f"Number of trials completed: {len(study.trials)}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print("="*50)

# Save best parameters
with open(f"{results_folder}best_params.txt", "w") as f:
    f.write("="*50 + "\n")
    f.write("BEST GNN HYPERPARAMETERS\n")
    f.write("="*50 + "\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nBest CV Macro-F1: {study.best_value:.4f}\n")
    f.write(f"Number of trials: {len(study.trials)}\n")
    f.write(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
    f.write("="*50 + "\n")

print("\nTraining final GNN with optimized parameters...")

# Clear any previous models
tf.keras.backend.clear_session()

# Build final adjacency matrices with best k_neighbors
print("Building graph structures from feature space...")
graph_tr = build_graph_tensor_from_knn(X_tr_np, k_neighbors=best_params['k_neighbors'])
graph_val = build_graph_tensor_from_knn(X_val_np, k_neighbors=best_params['k_neighbors'])

gnn_model = build_gnn(
    graph_tr.spec,
    best_params['hidden_dim'],
    best_params['num_layers'],
    num_classes,
    dropout=best_params['dropout'],
    l2_reg=best_params['l2_reg']
)

# Compute final class weights
classes_final = np.unique(y_tr)
cw_final = compute_class_weight('balanced', classes=classes_final, y=y_tr)
class_weights_final = {int(c): float(w) for c, w in zip(classes_final, cw_final)}

train_model(
    gnn_model, graph_tr, y_tr,
    graph_val, y_val,
    epochs=150,
    lr=best_params['lr'],
    patience=20,
    class_weights=class_weights_final,
    trial=None
)

print("Training complete")

# Save model (use SavedModel format for TF-GNN compatibility)
model_path = os.path.join(results_folder, "gnn_model")
gnn_model.save(model_path, save_format='tf')
print(f"\nSaved final model to {model_path}")

# Evaluate on validation set
evaluate_set_gnn(gnn_model, X_val_np, best_params['k_neighbors'], y_val,
                 "INTERNAL VALIDATION SET (Set 1 - 30%)", le,
                 save_file=f"{results_folder}validation_set_results.txt")

X_test2_np = X_test2_scaled.values.astype(np.float32)
X_test3_np = X_test3_scaled.values.astype(np.float32)

evaluate_set_gnn(gnn_model, X_test2_np, best_params['k_neighbors'], y_test2_enc,
                    "EXTERNAL TEST SET (Set 2)", le,
                    save_file=f"{results_folder}test_set2_results.txt")

evaluate_set_gnn(gnn_model, X_test3_np, best_params['k_neighbors'], y_test3_enc,
                    "EXTERNAL TEST SET (Set 3)", le,
                     save_file=f"{results_folder}test_set3_results.txt")



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
#   Performs post-hoc analysis of the best MLP IC model, including
#   class-wise SHAP feature importance plots, confusion matrices,
#   calibration curves, and Decision Curve Analysis across all datasets.

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from model_helper import get_sets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from sklearn.calibration import calibration_curve
from dcurves import dca, plot_graphs

np.random.seed(19)
tf.random.set_seed(19)

# ── Configuration ─────────────────────────────────────────────────────────────
# Parse arguments
name = "ic"
fs_method = "RFELASSO"
num_fs = "50"

reverse_dict = {
    "ENSG00000076242.16": "MLH1 (R)",
"ENSG00000178567.8": "EPM2AIP1 (R)",
"ENSG00000144214.10": "LYG1 (R)",
"ENSG00000231503.4": "PTMAP4 (R)",
"ENSG00000163584.18": "RPL22L1 (R)",
"ENSG00000184719.12": "RNLS (R)",
"ENSG00000154760.14": "SLFN13 (R)",
"ENSG00000198134.3": "AC007537.1 (R)",
"ENSG00000162639.16": "HENMT1 (R)",
"ENSG00000205502.4": "C2CD4B (R)",
"ENSG00000243181.2": "AC087343.1 (R)",
"ENSG00000057468.7": "MSH4 (R)",
"ENSG00000170468.8": "RIOX1 (R)",
"ENSG00000273305.1": "AC009237.15 (R)",
"ENSG00000287080.2": "H3FC (R)",
"ENSG00000179101.5": "AL590139.1 (R)",
"ENSG00000252421.1": "RNU6-1069P (R)",
"ENSG00000010932.17": "FMO1 (R)",
"ENSG00000272913.1": "AC009237.14 (R)",
"ENSG00000229689.3": "AC009237.3 (R)",
"ENSG00000277406.2": "AC245407.1 (R)",
"ENSG00000178458.5": "H3F3AP6 (R)",
"ENSG00000270938.1": "RAP2CP1 (R)",
"ENSG00000278071.1": "AL161669.3 (R)",
"ENSG00000115970.18": "THADA (R)",
"ENSG00000228223.3": "HCG11 (R)",
"ENSG00000254126.8": "CD8B2 (R)",
"ENSG00000272944.1": "AC079834.2 (R)",
"ENSG00000147889.18": "CDKN2A (R)",
"ENSG00000231084.2": "RPL22P24 (R)",
"ENSG00000119698.12": "PPP4R4 (R)",
"ENSG00000164002.11": "EXO5 (R)",
"ENSG00000165548.11": "TMEM63C (R)",
"ENSG00000164430.16": "CGAS (R)",
"ENSG00000116014.10": "KISS1R (R)",
"ENSG00000274372.5": "AC239803.4 (R)",
"ENSG00000273132.1": "AL355312.3 (R)",
"ENSG00000182247.10": "UBE2E2 (R)",
"ENSG00000242670.1": "RPL22P13 (R)",
"ENSG00000232412.1": "AL121601.1 (R)",
"ENSG00000220848.5": "RPS18P9 (R)",
"ENSG00000234636.2": "MED14OS (R)",
"ENSG00000267365.1": "KCNJ2-AS1 (R)",
"ENSG00000276900.1": "AC023157.3 (R)",
"ENSG00000246705.4": "H2AFJ (R)",
"ENSG00000277873.1": "AC131159.2 (R)",
"ENSG00000115598.10": "IL1RL2 (R)",
"ENSG00000167476.10": "JSRP1 (R)",
"ENSG00000185088.14": "RPS27L (R)",
"ENSG00000287431.1": "RENO1 (R)"
}

results_folder = f"results2601/mlp/{fs_method}_{name}_{num_fs}/"
shap_folder = "shap_ic_mlp_lasso50"
os.makedirs(shap_folder, exist_ok=True)

print(f"Loading data and model for: {name}, {fs_method}, {num_fs}")

# ── Load model and data ───────────────────────────────────────────────────────
model_path = os.path.join(results_folder, "mlp_model.keras")
mlp_model = tf.keras.models.load_model(model_path)

X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs)

X_tr_np = X_tr_scaled.values.astype(np.float32)
X_val_np = X_val_scaled.values.astype(np.float32)
X_test2_np = X_test2_scaled.values.astype(np.float32)
X_test3_np = X_test3_scaled.values.astype(np.float32)

# ── Functions ─────────────────────────────────────────────────────────────────
def display_names(feature_names):
    feature_names_display = [reverse_dict.get(name, name) for name in feature_names]
    for i, name in enumerate(feature_names_display):
        if " (R)" in name:
            continue
        if name.startswith("hsa-"):
            feature_names_display[i] = name + " (M)"
        else:
            feature_names_display[i] = name + " (D)"
    
    return feature_names_display

explainer = shap.Explainer(mlp_model, X_tr_scaled, seed=19)

def create_shap_plots(X_data, data_name, explainer, save_folder, feature_names, le):
    print(f"\nSHAP: {data_name}")
    shap_values_plot = explainer(X_data)
   
    n_classes = shap_values_plot.shape[2]
    class_names = le.classes_
    
    # Prepare feature names
    feature_names_display = display_names(feature_names)
    
    # Initialize importance dictionary for CSV
    importance_data = {'feature': feature_names_display}
    
    # Class-wise SHAP plots and collect importance
    for i in range(n_classes):
        class_label = le.inverse_transform([i])[0]
        
        # Get SHAP values for this class
        shap_vals_class = shap_values_plot[:, :, i]
        
        # Calculate mean absolute SHAP for this class
        class_importance = np.abs(shap_vals_class.values).mean(axis=0)
        importance_data[f'class_{class_label}'] = class_importance
        
        # Top 20
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_class, X_data, feature_names=feature_names_display, 
                          plot_type="dot", show=False, max_display=20, rng=19)
        plt.title(f"SHAP - {data_name} - Class: {class_label}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"shap20_{data_name.replace(' ', '_')}_{class_label}.png"), 
                    dpi=300, bbox_inches='tight')

        plt.close()
        
        # Top 10
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_class, X_data, feature_names=feature_names_display,
                          plot_type="dot", show=False, max_display=10, rng=19)
        plt.title(f"SHAP - {data_name} - Class: {class_label}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"shap10_{data_name.replace(' ', '_')}_{class_label}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate mean importance across all classes
    all_shap = np.abs(shap_values_plot.values).mean(axis=0)  # Mean across samples
    mean_importance = all_shap.mean(axis=1)  # Mean across classes
    importance_data['mean_importance'] = mean_importance
    
    # Create DataFrame and sort by mean importance
    shap_importance_df = pd.DataFrame(importance_data)
    shap_importance_df = shap_importance_df.sort_values('mean_importance', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(save_folder, f"shap_feature_importance_{data_name.replace(' ', '_')}.csv")
    shap_importance_df.to_csv(csv_path, index=False)
    print(f"SHAP importance saved to: {csv_path}")
    

    # Mean SHAP plot - Top 20
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_plot, X_data, feature_names=feature_names_display,
                      plot_type="bar", show=False, max_display=20, rng=19)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Replace legend labels with real class names
    new_labels = [
        class_names[i] if i < len(class_names) else lab
        for i, lab in enumerate(labels)
    ]

    ax.legend(handles, new_labels, title="Class")
    plt.title(f"SHAP (Mean Across Classes) - {data_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"shap20_mean_{data_name.replace(' ', '_')}.png"), 
                dpi=300, bbox_inches='tight')

    plt.close()
    
    # Mean SHAP plot - Top 10
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_plot, X_data, feature_names=feature_names_display,
                      plot_type="bar", show=False, max_display=10, rng=19)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Replace legend labels with real class names
    new_labels = [
        class_names[i] if i < len(class_names) else lab
        for i, lab in enumerate(labels)
    ]

    ax.legend(handles, new_labels, title="Class")
    plt.title(f"SHAP (Mean Across Classes) - {data_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"shap10_mean_{data_name.replace(' ', '_')}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()




def create_confusion_matrix(y_true, y_pred, data_name, save_folder, le):
    print(f"Confusion Matrix: {data_name}")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {data_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"confusion_matrix_{data_name.replace(' ', '_')}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_calibration_curve(y_true, y_pred_proba, data_name, save_folder, le):
    n_classes = y_pred_proba.shape[1]
    class_labels = le.classes_
    y_true = le.inverse_transform(y_true)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    plt.figure(figsize=(10, 7))
    
    brier_scores = {}
    for i, class_label in enumerate(class_labels):
        y_binary = (y_true == class_label).astype(int)
        prob_class = y_pred_proba[:, i]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_binary, prob_class, n_bins=10, strategy='quantile')
        
        # Calculate Brier score for this class
        brier = brier_score_loss(y_binary, prob_class)
        brier_scores[f'Class {class_label}'] = brier
        
        # Plot
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color=colors[i], label=f'Class {class_label} (Brier: {brier:.3f})')
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfectly calibrated')
    
    # Calculate mean Brier score
    mean_brier = np.mean(list(brier_scores.values()))
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f"Calibration Curves - {data_name}\nMean Brier Score: {mean_brier:.4f}", 
                fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(save_folder, f"calibration_{data_name.replace(' ', '_')}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return brier_scores


def create_decision_curve(y_true, y_pred_proba, data_name, save_folder, le):
    """
    Create decision curve analysis using the dcurves library.
    
    For multi-class problems, creates separate DCA plots for each class
    using a one-vs-rest approach.
    """
    n_classes = y_pred_proba.shape[1]
    class_labels = le.classes_
    y_true_labels = le.inverse_transform(y_true)
    
    for i, class_label in enumerate(class_labels):
        # Create binary outcome: current class vs all others
        y_binary = (y_true_labels == class_label).astype(int)
        prob_class = y_pred_proba[:, i]
        
        prevalence = np.mean(y_binary)
        
        # Create DataFrame for dcurves
        df_dca = pd.DataFrame({
            'outcome': y_binary,
            'pred_prob': prob_class
        })
        
        max_threshold = min(0.8, max(0.35, prevalence * 3))
        dca_result = dca(
            data=df_dca,
            outcome='outcome',
            modelnames=['pred_prob'],
            thresholds=np.arange(0, max_threshold, 0.01)
        )
        
        plt.switch_backend('Agg')
        
        fig = plt.figure(figsize=(10, 7))
        plot_graphs(
            plot_df=dca_result,
            graph_type='net_benefit',
            y_limits=[-0.05, max(0.25, prevalence * 1.2)],
            color_names=['#1f77b4', '#d62728', '#2ca02c'],
            smooth_frac=0.5,
        )
        
        # Update legend labels
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        
        # Replace legend labels with custom names
        new_labels = []
        for label in labels:
            if 'All' in label or label == 'all':
                new_labels.append('Treat All')
            elif 'None' in label or label == 'none':
                new_labels.append('Treat None')
            else:
                # This is the model prediction line
                new_labels.append(f'Treat {class_label}')
        
        ax.legend(handles, new_labels, loc='best', fontsize=11)
        
        # Add title
        plt.title(f'Decision Curve Analysis - {data_name}\nClass: {class_label} (Prevalence: {prevalence*100:.2f}%)',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_folder, f"dca_{data_name.replace(' ', '_')}_{class_label}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        print(f"DCA saved: {save_path} - Prevalence = {prevalence*100:.2f}%")
    
    print(f"Decision curve analysis completed for {data_name}")



def analyze_dataset(X_data, y_true, data_name, explainer, save_folder, le, feature_names):
    
    # SHAP
    create_shap_plots(X_data, data_name, explainer, save_folder, feature_names, le)
    
    # Predictions
    logits = mlp_model.predict(X_data, verbose=0)
    probs = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(probs, axis=1)
    
    # Confusion Matrix
    create_confusion_matrix(y_true, y_pred, data_name, save_folder, le)
    
    # Calibration Curve
    create_calibration_curve(y_true, probs, data_name, save_folder, le)
    
    # Decision Curve
    create_decision_curve(y_true, probs, data_name, save_folder, le)
    


# ── Main ──────────────────────────────────────────────────────────────────────
# Get feature names
feature_names = X_val_scaled.columns.tolist()

# Analyze each dataset
analyze_dataset(X_val_np, y_val.values, "Validation Set", explainer, shap_folder, le, feature_names)
analyze_dataset(X_test2_np, y_test2_enc, "Test Set 2", explainer, shap_folder, le, feature_names)
analyze_dataset(X_test3_np, y_test3_enc, "Test Set 3", explainer, shap_folder, le, feature_names)

# Combined test sets
X_test_combined = np.vstack([X_test2_np, X_test3_np])
y_test_combined = np.concatenate([y_test2_enc, y_test3_enc])
analyze_dataset(X_test_combined, y_test_combined, "Test Set 2 and 3 Combined", 
                explainer, shap_folder, le, feature_names)

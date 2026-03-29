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
#   Performs post-hoc analysis of the best LightGBM MSI model, including
#   SHAP feature importance plots, confusion matrices, calibration curves,
#   and Decision Curve Analysis (DCA) across validation and test sets.

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import shap
from model_helper import get_sets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from sklearn.calibration import calibration_curve
from dcurves import dca, plot_graphs
import seaborn as sns
import lightgbm

np.random.seed(19)

# ── Configuration ─────────────────────────────────────────────────────────────
name = "msi"
fs_method = "RFESVM"
num_fs = "20"

reverse_dict = {
    "ENSG00000179101.5": "AL590139.1 (R)",
    "ENSG00000163584.18": "RPL22L1 (R)",
    "ENSG00000246705.4": "H2AFJ (R)",
    "ENSG00000057468.7": "MSH4 (R)",
    "ENSG00000243181.2": "AC087343.1 (R)",
    "ENSG00000205502.4": "C2CD4B (R)",
    "ENSG00000231084.2": "RPL22P24 (R)",
    "ENSG00000231503.4": "PTMAP4 (R)",
    "ENSG00000178458.5": "H3F3AP6 (R)",
    "ENSG00000144214.10": "LYG1 (R)",
    "ENSG00000076242.16": "MLH1 (R)",
    "ENSG00000162639.16": "HENMT1 (R)",
    "ENSG00000130005.13": "GAMT (R)",
    "ENSG00000184719.12": "RNLS (R)",
    "ENSG00000149972.11": "CNTN5 (R)",
    "ENSG00000178567.8": "EPM2AIP1 (R)",
    "ENSG00000198134.3": "AC007537.1 (R)",
    "ENSG00000170468.8": "RIOX1 (R)",
    "ENSG00000229385.2": "AC010975.1 (R)",
    "ENSG00000277406.2": "AC245407.1 (R)"
}

results_folder = f"results2601/lgbm/{fs_method}_{name}_{num_fs}/"
shap_folder = "shap_msi_lgbm_svm20"
os.makedirs(shap_folder, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
model_path = os.path.join(results_folder, "lgbm_model.pkl")
lgbm_model = joblib.load(model_path)

#plot feature importance
df_feature_importance = (
    pd.DataFrame({
        'feature': lgbm_model.feature_name_,
        'importance': lgbm_model.feature_importances_,
    })
    .sort_values('importance', ascending=False)
)
#top-20
df_feature_importance = df_feature_importance.head(20)
#apply reverse dict to feature names
df_feature_importance['feature_display'] = df_feature_importance['feature'].map(reverse_dict).fillna(df_feature_importance['feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature_display', data=df_feature_importance, palette='viridis')
plt.title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(shap_folder, "inbuilt_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
sys.exit()


X_tr_scaled, y_tr, X_val_scaled, y_val, X_test2_scaled, y_test2_enc, X_test3_scaled, y_test3_enc, le = get_sets(name, fs_method, num_fs)

# Check label encoding
print("\n" + "="*60)
print("LABEL ENCODING CHECK:")
print(f"Classes: {le.classes_}")
print(f"Label 0 = {le.classes_[0]}")
print(f"Label 1 = {le.classes_[1]}")
print("="*60 + "\n")

explainer = shap.Explainer(lgbm_model, X_tr_scaled)


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

def create_shap_plots(X_data, data_name, explainer, save_folder, feature_names):
    shap_values = explainer(X_data)


    #To change the class
    shap_values.values = -shap_values.values
    shap_values.base_values = 1 - shap_values.base_values

    feature_names = display_names(feature_names)
    
    # Top 20
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_data, plot_type="dot", show=False, max_display=20, rng=19, feature_names=feature_names)
    plt.title(f"SHAP - {data_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"shap20_{data_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top 10
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_data, plot_type="dot", show=False, max_display=10, rng=19, feature_names=feature_names)
    plt.title(f"SHAP - {data_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"shap10_{data_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    #save shap values to csv
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': np.abs(shap_values.values).mean(axis=0)
    })

    # Sort by mean importance (descending)
    shap_importance = shap_importance.sort_values('mean_importance', ascending=False)

    # Save to CSV
    shap_importance.to_csv(os.path.join(save_folder, f"shap_feature_importance_{data_name.replace(' ', '_')}.csv"), index=False)
    
    return shap_values


def create_confusion_matrix(y_true, y_pred, data_name, save_folder, le):
    print(f"Confusion Matrix: {data_name}")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {data_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"confusion_matrix_{data_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_calibration_curve(y_true, y_pred_proba, data_name, save_folder):
    print(f"Calibration Curve: {data_name}")
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='LightGBM')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f"Calibration Curve - {data_name}\nBrier Score: {brier:.4f}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"calibration_{data_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return brier


def create_decision_curve(y_true, y_pred_proba, data_name, save_folder):
    """
    Create decision curve analysis using the dcurves library.
    For binary classification - predicting MSI-H (class 0).
    
    y_pred_proba should be the probability of MSI-H (class 0).
    y_true should contain 0 (MSI-H) and 1 (MSS).
    """
    print(f"Decision Curve: {data_name}")
    
    # Convert to binary: 1 if MSI-H (class 0), 0 if MSS (class 1)
    # This makes the DCA interpret the outcome correctly
    y_binary = (y_true == 0).astype(int)
    prevalence = np.mean(y_binary)
    
    # Create DataFrame for dcurves
    df_dca = pd.DataFrame({
        'outcome': y_binary,
        'pred_prob': y_pred_proba
    })
    
    # Determine threshold range based on prevalence
    max_threshold = min(0.8, max(0.35, prevalence * 3))
    
    # Run decision curve analysis
    dca_result = dca(
        data=df_dca,
        outcome='outcome',
        modelnames=['pred_prob'],
        thresholds=np.arange(0, max_threshold, 0.01)
    )
    
    # Set backend to prevent display
    plt.switch_backend('Agg')
    
    # Create plot
    fig = plt.figure(figsize=(10, 7))
    plot_graphs(
        plot_df=dca_result,
        graph_type='net_benefit',
        y_limits=[-0.05, max(0.3, prevalence * 1.5)],
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
            new_labels.append('Treat MSI-H')
    
    ax.legend(handles, new_labels, loc='best', fontsize=11)
    
    # Add title
    plt.title(f'Decision Curve Analysis - {data_name}\nPrevalence: {prevalence*100:.2f}%',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_folder, f"dca_{data_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"DCA saved: {save_path} - Prevalence = {prevalence:.3f}")


def analyze_dataset(X_data, y_true, data_name, explainer, save_folder, le, feature_names=None):
    shap_values = create_shap_plots(X_data, data_name, explainer, save_folder, feature_names)
    
    # Predictions
    y_pred = lgbm_model.predict(X_data)
    
    # Get probability of MSI-H (class 0)
    # Since le.classes_ = ['MSI-H', 'MSS'], class 0 is MSI-H
    y_pred_proba = lgbm_model.predict_proba(X_data)[:, 0]
    
    print(f"\n{data_name}:")
    print(f"  Probability range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    print(f"  Mean probability: {y_pred_proba.mean():.3f}")
        
    # Confusion Matrix
    create_confusion_matrix(y_true, y_pred, data_name, save_folder, le)
    
    # Calibration Curve
    brier = create_calibration_curve(y_true, y_pred_proba, data_name, save_folder)
    
    # Decision Curve
    create_decision_curve(y_true, y_pred_proba, data_name, save_folder)
    

feature_names = X_val_scaled.columns.tolist()

# ── Main ──────────────────────────────────────────────────────────────────────
# Analyze each dataset
analyze_dataset(X_val_scaled, y_val, "Validation Set", explainer, shap_folder, le, feature_names)
analyze_dataset(X_test2_scaled, y_test2_enc, "Test Set-2", explainer, shap_folder, le, feature_names)
analyze_dataset(X_test3_scaled, y_test3_enc, "Test Set-3", explainer, shap_folder, le, feature_names)

# Combined test sets
X_test_combined = pd.concat([X_test2_scaled, X_test3_scaled], axis=0, ignore_index=True)
y_test_combined = pd.concat([pd.Series(y_test2_enc), pd.Series(y_test3_enc)], axis=0, ignore_index=True)
analyze_dataset(X_test_combined, y_test_combined, "Test Set 2 and 3 Combined", explainer, shap_folder, le, feature_names)



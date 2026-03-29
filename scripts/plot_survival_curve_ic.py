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
#   Generates Kaplan-Meier survival curves stratified by Genomic Subtype
#   (IC classes) and runs univariate/multivariate Cox proportional-hazards
#   models for the top SHAP-selected features across all three datasets.

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

# ── Configuration ─────────────────────────────────────────────────────────────
REVERSE_DICT = {
    "PPP4R4": "ENSG00000119698.12",
    "MSH4": "ENSG00000057468.7",
    "MLH1": "ENSG00000076242.16",
    "EPM2AIP1": "ENSG00000178567.8",
}

COL_DICT = {
"ENSG00000119698.12": "PPP4R4 (RNA)",
"ENSG00000057468.7": "MSH4 (RNA)",
"ENSG00000076242.16": "MLH1 (RNA)",
"ENSG00000178567.8": "EPM2AIP1 (RNA)",
"hsa-mir-378a": "hsa-mir-378a (miRNA)",
}

# ── Functions ─────────────────────────────────────────────────────────────────
def prepare_survival_data(set_name):
    survival_df = pd.read_csv(f"data_extraction/metadata/{set_name}_labeled_with_survival.csv", index_col=0)
    survival_df.dropna(subset=["Genomic_Subtype"], inplace=True)
    survival_df = survival_df[survival_df["Vital_Status"].isin(["Alive", "Dead"])]
    survival_df['event'] = (survival_df['Vital_Status'] == 'Dead').astype(int)
    survival_df['time'] = np.where(
        survival_df['Vital_Status'] == 'Dead',
        survival_df['Days_To_Death'],
        survival_df['Days_To_Last_Follow_Up']
    )
    survival_df = survival_df.dropna(subset=['time', 'event'])
    survival_df = survival_df[survival_df['time'] > 0]
    return survival_df


def plot_km_by_genomic_subtype(survival_df, output_dir):
    y = np.array(
        [(bool(e), t) for e, t in zip(survival_df['event'], survival_df['time'])],
        dtype=[('event', bool), ('time', float)]
    )
    
    plt.figure(figsize=(6, 4.5))
    genomic_subtypes = survival_df['Genomic_Subtype'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(genomic_subtypes)))
    
    for i, subtype in enumerate(genomic_subtypes):
        mask = survival_df['Genomic_Subtype'] == subtype
        time, survival_prob = kaplan_meier_estimator(y['event'][mask], y['time'][mask])
        n_patients = mask.sum()
        plt.step(time, survival_prob, where="post", linewidth=2, 
                 label=f'{subtype} (n={n_patients})', color=colors[i])
    
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.title('Kaplan-Meier Survival Curves by Genomic Subtype', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kaplan_meier_genomic_subtype.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    chisq, pvalue = compare_survival(y, survival_df['Genomic_Subtype'].values)
    print(f"Log-rank p={pvalue:.4f}, Chi-sq={chisq:.4f}")
    return y


def load_feature_data(set_name, survival_df, top_5_features):
    rna = [f.replace(" (R)", "") for f in top_5_features if "(R)" in f]
    mirna = [f.replace(" (M)", "") for f in top_5_features if "(M)" in f]
    methyl = [f.replace(" (D)", "") for f in top_5_features if "(D)" in f]
    
    if rna:
        rna = [REVERSE_DICT.get(gene, gene) for gene in rna]
        feature_df = pd.read_csv(f"data_extraction/data/{set_name}_rna_model.csv", index_col=0)[rna + ["Genomic_Subtype"]]
        feature_df = feature_df.join(survival_df[['time', 'event']], on=feature_df.index, how='left')
        feature_df.dropna(subset=['time', 'event', 'Genomic_Subtype'], inplace=True)
        if 'key_0' in feature_df.columns:
            feature_df.drop(columns=['key_0'], inplace=True)
    
    if mirna:
        mirna_df = pd.read_csv(f"data_extraction/data/{set_name}_mirna_model.csv", index_col=0)[mirna]
        feature_df = feature_df.join(mirna_df, on=feature_df.index, how='left')
    
    if methyl:
        methyl_df = pd.read_csv(f"data_extraction/data/{set_name}_methylation_model.csv", index_col=0)[methyl]
        feature_df = feature_df.join(methyl_df, on=feature_df.index, how='left')
    
    feature_df.rename(columns=COL_DICT, inplace=True)
    return feature_df


def run_survival_analysis(feature_df, genes, output_dir):
    y = Surv.from_dataframe(event="event", time="time", data=feature_df)
    
    time, survival = kaplan_meier_estimator(y["event"], y["time"])
    
    X_cox = feature_df[genes].values
    cox_model = CoxPHSurvivalAnalysis()
    cox_model.fit(X_cox, y)
    
    print(f"Cox C-index: {cox_model.score(X_cox, y):.4f}")
    cox_results = pd.DataFrame({
        'Gene': genes,
        'Coefficient': cox_model.coef_,
        'Hazard_Ratio': np.exp(cox_model.coef_)
    })
    cox_results.to_csv(os.path.join(output_dir, 'cox_model_results.csv'), index=False)
    
    feature_df_encoded = feature_df.copy()
    feature_df_encoded['Genomic_Subtype_Encoded'] = feature_df_encoded['Genomic_Subtype'].astype('category').cat.codes
    X_multi = feature_df_encoded[genes + ['Genomic_Subtype_Encoded']].values
    
    cox_multi = CoxPHSurvivalAnalysis()
    cox_multi.fit(X_multi, y)
    
    print(f"Multivariate Cox C-index: {cox_multi.score(X_multi, y):.4f}")
    covariate_names = genes + ['Genomic_Subtype_Encoded']
    multi_results = pd.DataFrame({
        'Covariate': covariate_names,
        'Coefficient': cox_multi.coef_,
        'Hazard_Ratio': np.exp(cox_multi.coef_)
    })
    multi_results.to_csv(os.path.join(output_dir, 'cox_multivariate_results.csv'), index=False)
    
    return y, multi_results


def plot_gene_km_curves(feature_df, genes, y, multi_results, output_dir):
    for gene in genes:
        median_expr = feature_df[gene].median()
        groups = feature_df[gene] >= median_expr
        
        plt.figure(figsize=(6, 4.5))
        for label, mask in {"High": groups, "Low": ~groups}.items():
            time, surv = kaplan_meier_estimator(y["event"][mask], y["time"][mask])
            n_samples = mask.sum()
            plt.step(time, surv, where="post", label=f"{label} (n={n_samples})")

        hazard_ratio = multi_results.loc[multi_results['Covariate'] == gene, 'Hazard_Ratio'].values[0]
        # Place HR box at the bottom
        plt.text(0.83, 0.17, f'HR={hazard_ratio:.4f}', transform=plt.gca().transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7), ha='center')

        plt.grid(alpha=0.3)
        plt.ylim([0, 1])
        plt.xlim([0, None])
        plt.title(f"{gene}")
        plt.xlabel("Time (Years)")
        plt.ylabel("Survival Probability")
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'km_by_{gene.replace(" ", "_").replace("(", "").replace(")", "")}_expression.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def process_dataset(set_name, top_5_features):
    print(f"\n{'='*60}")
    print(f"Processing {set_name}")
    print(f"{'='*60}")
    
    output_dir = f"survival_ic_analysis/{set_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    survival_df = prepare_survival_data(set_name)
    plot_km_by_genomic_subtype(survival_df, output_dir)
    feature_df = load_feature_data(set_name, survival_df, top_5_features)
    genes = [COL_DICT[x] for x in COL_DICT if COL_DICT[x] in feature_df.columns]
    y, multi_results = run_survival_analysis(feature_df, genes, output_dir)
    plot_gene_km_curves(feature_df, genes, y, multi_results, output_dir)


# ── Main ──────────────────────────────────────────────────────────────────────
imp_fs = pd.read_csv("shap_ic_mlp_lasso50/shap_feature_importance_Test_Set_2_and_3_Combined.csv")
top_5_features = imp_fs['feature'].head(5).tolist()
print("Top 5 features for survival analysis:", top_5_features)

os.makedirs("survival_ic_analysis", exist_ok=True)

for set_name in ['set1', 'set2', 'set3']:
    process_dataset(set_name, top_5_features)


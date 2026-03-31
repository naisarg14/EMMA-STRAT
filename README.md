# EMMA-STRAT

## A Multi-Omics Based Machine Learning Framework for Stratification of Endometrial Carcinoma Molecular Subtypes and MSI Status

**Authors:** Naisarg Patel, Andres Salumets, Vijayachitra Modhukur

**License:** GNU General Public License v3.0 — see [LICENSE.txt](LICENSE.txt)

---

## Overview

EMMA-STRAT is a multi-omics machine learning framework for stratifying endometrial carcinoma (EC) patients by:

- **Genomic subtype (IC):** POLE, MSI, CNL, CNH
- **MSI status:** MSS vs MSI-H

The framework integrates three omics layers — **DNA methylation**, **RNA-seq**, and **miRNA** — from TCGA-UCEC data, covering the full pipeline from data retrieval and preprocessing through feature selection, model training, post-hoc analysis, and survival visualization.

---

## Repository Structure

```
scripts/
│
├── Data Retrieval & Preprocessing
│   ├── query_gdc_files.py          Query GDC API for file IDs by case ID and data type
│   ├── download_gdc_files.py       Download and merge multi-omics files from GDC API
│   ├── extract.py                  Download UCEC data from LinkedOmics/GDC; merge clinical labels
│   ├── methylation.py              Variance filtering, NA removal, beta-to-M-value conversion
│   ├── methylation.R               Aggregate CpG probes to gene-level beta values (Illumina arrays)
│   ├── mirna.py                    Build miRNA count matrices; CPM + log2 normalisation
│   ├── rna.py                      Extract TPM matrices from RNA-seq files; log2(TPM+1) transformation
│   └── find_common.py              Identify common features and samples across omics datasets
│
├── Feature Selection
│   ├── FS_svm.py                   RFE with linear SVM
│   ├── FS_rf.py                    RFE with Random Forest
│   ├── FS_lasso.py                 LASSO (L1 logistic regression)
│   └── FS_anova.py                 ANOVA F-test (SelectKBest)
│
├── Model Training — Multi-Omics
│   ├── model_helper.py             Shared data-loading and evaluation utilities (used by all models)
│   ├── run_models.py               Orchestrate all model training scripts via subprocess
│   ├── lgbm.py                     LightGBM with Optuna hyperparameter tuning
│   ├── mlp.py                      MLP neural network with Optuna tuning
│   ├── gnn.py                      Graph Neural Network (TF-GNN / mt_albis) with Optuna tuning
│   ├── svm.py                      SVM with Optuna tuning
│   ├── rf.py                       Random Forest with Optuna tuning
│   └── knn.py                      K-Nearest Neighbors with Optuna tuning
│
├── Model Training — Single-Omics
│   ├── single_lgbm.py              Single-omics LightGBM (iterates over miRNA / RNA / methylation)
│   ├── single_mlp.py               Single-omics MLP
│   ├── single_gnn.py               Single-omics GNN
│   ├── single_svm.py               Single-omics SVM
│   ├── single_rf.py                Single-omics Random Forest
│   └── single_knn.py               Single-omics KNN
│
├── Results & Analysis
│   ├── extract_results.py          Aggregate multi-omics model results into summary CSVs
│   ├── extract_single.py           Aggregate single-omics model results into summary CSVs
│   ├── analysis_lgbm.py            LightGBM post-hoc: SHAP, confusion matrix, calibration, DCA
│   ├── analysis_mlp.py             MLP post-hoc: SHAP, confusion matrix, calibration, DCA
│   ├── bootstrap_ci_lgbm_msi.py    Bootstrap 95% CIs for LightGBM MSI model metrics
│   └── bootstrap_ci_mlp_ic.py      Bootstrap 95% CIs for MLP IC model metrics
│
├── Visualisation
│   ├── plot_survival_curve_msi.py  Kaplan-Meier + Cox survival curves (MSI stratification)
│   ├── plot_survival_curve_ic.py   Kaplan-Meier + Cox survival curves (IC stratification)
│   └── venn_diagram_maker.py       Venn diagrams of feature overlap across IC and MSI models
│
├── Benchmarking
│   └── prepare_flexynesis_data.py  Prepare multi-omics data in Flexynesis-compatible format
│
└── feature_selection/              Saved selected feature lists (per method / omics / k)
    ├── RFESVM_msi/
    ├── RFESVM_ic/
    ├── RFELASSO_ic/
    ├── RFERF_ic/
    ├── ANOVA_ic/
    └── ...
```

---

## Pipeline Overview

```
1. Data Retrieval
   query_gdc_files.py → download_gdc_files.py → extract.py

2. Preprocessing
   methylation.py / methylation.R / mirna.py / rna.py → find_common.py

3. Feature Selection  (run per omics type, per label, per k)
   FS_svm.py / FS_rf.py / FS_lasso.py / FS_anova.py
              ↓
   feature_selection/<METHOD>_<LABEL>/selected_<omics>_<METHOD>_<k>.txt

4. Model Training
   run_models.py → lgbm.py / mlp.py / gnn.py / svm.py / rf.py / knn.py  (multi-omics)
                 → single_lgbm.py / single_mlp.py / ...                  (single-omics)

5. Results Aggregation
   extract_results.py / extract_single.py

6. Post-Hoc Analysis
   analysis_lgbm.py / analysis_mlp.py
   bootstrap_ci_lgbm_msi.py / bootstrap_ci_mlp_ic.py

7. Visualisation
   plot_survival_curve_msi.py / plot_survival_curve_ic.py / venn_diagram_maker.py
```

---

## Requirements

### Python (≥ 3.9)

```
numpy
pandas
scikit-learn
lightgbm
tensorflow
tensorflow-gnn
optuna
shap
matplotlib
seaborn
scipy
statsmodels
joblib
imbalanced-learn
sksurv
dcurves
tqdm
requests
openpyxl
```

Install all at once:

```bash
pip install numpy pandas scikit-learn lightgbm tensorflow tensorflow-gnn optuna \
            shap matplotlib seaborn scipy statsmodels joblib imbalanced-learn \
            scikit-survival dcurves tqdm requests openpyxl
```

### R (≥ 4.2)

```r
install.packages(c("BiocManager", "ggplot2", "tidyverse"))
BiocManager::install(c("TCGAbiolinks", "SummarizedExperiment", "minfi",
                       "IlluminaHumanMethylation450kanno.ilmn12.hg19"))
```

---

## Data

- **Cohort:** TCGA-UCEC (The Cancer Genome Atlas — Uterine Corpus Endometrial Carcinoma)
- **Omics layers:** DNA methylation (Illumina 450k), RNA-seq (TPM), miRNA-seq (CPM)
- **Labels:** Molecular subtype (`IntegrativeCluster`: POLE / MSI / CNL / CNH) and MSI status (`MSS` / `MSI-H`)
- **Access:** [GDC Data Portal](https://portal.gdc.cancer.gov/) — raw data are not distributed with this repository

Use `extract.py` and `download_gdc_files.py` to retrieve the raw data before running the pipeline.

---

## Usage

### 1. Retrieve and preprocess data

```bash
python scripts/query_gdc_files.py
python scripts/download_gdc_files.py
python scripts/extract.py
python scripts/methylation.py
python scripts/mirna.py
python scripts/rna.py
python scripts/find_common.py
Rscript scripts/methylation.R
```

### 2. Feature selection

```bash
# Arguments: <data_file> <label> <n_features>
python scripts/FS_svm.py   methylation_data.csv msi 20
python scripts/FS_lasso.py rna_data.csv         ic  50
python scripts/FS_rf.py    mirna_data.csv        msi 10
python scripts/FS_anova.py methylation_data.csv  ic  100
```

### 3. Train models

```bash
# Run all model/feature/label combinations
python scripts/run_models.py lgbm.py mlp.py gnn.py svm.py rf.py knn.py

# Or run a single configuration: <label> <fs_method> <n_features>
python scripts/lgbm.py msi RFESVM 20
python scripts/mlp.py  ic  RFELASSO 50
```

### 4. Aggregate results

```bash
python scripts/extract_results.py   # multi-omics
python scripts/extract_single.py    # single-omics
```

### 5. Post-hoc analysis and visualisation

```bash
python scripts/analysis_lgbm.py
python scripts/analysis_mlp.py
python scripts/bootstrap_ci_lgbm_msi.py
python scripts/bootstrap_ci_mlp_ic.py
python scripts/plot_survival_curve_msi.py
python scripts/plot_survival_curve_ic.py
python scripts/venn_diagram_maker.py
```


---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE.txt](LICENSE.txt) for the full terms.

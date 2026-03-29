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
#   Aggregates single-omics model results by iterating over all combinations
#   of model, feature selection method, label, feature count, and omics type.
#   Reads per-run result text files and writes consolidated CSV tables.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import subprocess

# ── Configuration ─────────────────────────────────────────────────────────────
codes = ["lgbm", "knn", "rf", "svm", "mlp", "gnn"] #["knn.py", "svm.py", "rf.py", "lgbm.py", "mlp.py"]
features = [10, 20, 50, 100, 150, 200]
fs_method = ["RFERF", "RFESVM", "RFELASSO", "ANOVA"]
labels = ["msi", "ic"]
omics = ["mirnas", "rna", "methyl"]


def smart_round(x):
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return f"{f:.4f}"
    except Exception:
        return x

reverse_dict_label = {
    "ic": "Genomic Subtype",
    "msi": "MSI Status",
    "nt": "Tissue Type",
}

reverse_dict_model = {
    "knn": "KNN",
    "svm": "Linear SVM",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "gnn": "GNN",
}


reverse_dict_fs = {
    "RFERF": "Random Forest",
    "RFESVM": "SVM",
    "RFELASSO": "LASSO",
    "ANOVA": "ANOVA F-test",
}

reverse_dict_data_type = {
    "mirnas": "miRNA",
    "rna": "RNA Seq",
    "methyl": "Methylation",
}

# ── Functions ─────────────────────────────────────────────────────────────────
def extract_results():
    import pandas as pd
    df = pd.DataFrame()
    summary_df = pd.DataFrame()
    count = 1
    not_found = []
    total = len(codes) * len(features) * len(fs_method) * len(labels) * len(omics)
    for code in codes:
        code = code.replace(".py", "")
        for fs in fs_method:
            for label in labels:
                for num in features:
                    try:
                        result_folder = f"single/{code}/{fs}_{label}_{num}/"
                        common_data = {
                            "model": code,
                            "fs_method": fs,
                            "label": label,
                            "num_features": num,
                        }
                        summary_dict = common_data.copy()
                        for type in omics:
                            msg = f"Extracting results: {count}/{total} ({((count-1)/total)*100:.2f}%)"
                            print(msg.ljust(80), end="\r", flush=True)
                            count += 1
                            summary_dict = common_data.copy()
                            summary_dict["data_type"] = type
                            with open(os.path.join(os.path.join(result_folder, type), "validation_set_results.txt"), "r") as val:
                                val_lines = val.readlines()
                                results = common_data.copy()
                                results["set"] = "Validation (30% TCGA)"
                                for line in val_lines:
                                    line = line.strip()
                                    if line.startswith("Balanced accuracy:"):
                                        results["balanced_accuracy"] = line.split(":")[1].strip()
                                        summary_dict["val_balanced_accuracy"] = line.split(":")[1].strip()
                                    elif line.startswith("Macro F1:"):
                                        results["macro_f1"] = line.split(":")[1].strip()
                                        summary_dict["val_macro_f1"] = line.split(":")[1].strip()
                                    elif line.startswith("Class"):
                                        parts = line.replace("Class", "").split(":")
                                        if len(parts) == 2:
                                            results[f"AUC_{parts[0].strip()}"] = parts[1].strip()
                                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

                            if label == "nt":
                                summary_df = pd.concat([summary_df, pd.DataFrame([summary_dict])], ignore_index=True)
                                continue
                            with open(os.path.join(os.path.join(result_folder, type), "test_set2_results.txt"), "r") as val:
                                test2_lines = val.readlines()
                                results = common_data.copy()
                                results["set"] = "Test Set 2 (CPTAC)"
                                for line in test2_lines:
                                    line = line.strip()
                                    if line.startswith("Balanced accuracy:"):
                                        results["balanced_accuracy"] = line.split(":")[1].strip()
                                        summary_dict["test2_balanced_accuracy"] = line.split(":")[1].strip()
                                    elif line.startswith("Macro F1:"):
                                        results["macro_f1"] = line.split(":")[1].strip()
                                        summary_dict["test2_macro_f1"] = line.split(":")[1].strip()
                                    elif line.startswith("Class"):
                                        parts = line.replace("Class", "").split(":")
                                        if len(parts) == 2:
                                            results[f"AUC_{parts[0].strip()}"] = parts[1].strip()
                                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

                            with open(os.path.join(os.path.join(result_folder, type), "test_set3_results.txt"), "r") as val:
                                test3_lines = val.readlines()
                                results = common_data.copy()
                                results["set"] = "Test Set 3 (CPTAC-Independent)"
                                for line in test3_lines:
                                    line = line.strip()
                                    if line.startswith("Balanced accuracy:"):
                                        results["balanced_accuracy"] = line.split(":")[1].strip()
                                        summary_dict["test3_balanced_accuracy"] = line.split(":")[1].strip()
                                    elif line.startswith("Macro F1:"):
                                        results["macro_f1"] = line.split(":")[1].strip()
                                        summary_dict["test3_macro_f1"] = line.split(":")[1].strip()
                                    elif line.startswith("Class"):
                                        parts = line.replace("Class", "").split(":")
                                        if len(parts) == 2:
                                            results[f"AUC_{parts[0].strip()}"] = parts[1].strip()
                                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
                            summary_df = pd.concat([summary_df, pd.DataFrame([summary_dict])], ignore_index=True)
                    except Exception as e:
                        not_found.append(result_folder)
                        continue


    #in df, if number then round to 4 decimal places
    df = df.map(smart_round)
    #replace model, fs_method, label with full names
    df["model"] = df["model"].map(reverse_dict_model)
    df["fs_method"] = df["fs_method"].map(reverse_dict_fs)
    df["label"] = df["label"].map(reverse_dict_label)


    #rename columns
    df = df.rename(columns={
        "model": "Model",
        "fs_method": "Feature Selection Method",
        "label": "Prediction Task",
        "num_features": "Number of Features",
        "set": "Dataset",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro F1 Score",
    })
    df.to_csv("model_single_results.csv",index=False)


    summary_df = summary_df.map(smart_round)
    summary_df["model"] = summary_df["model"].map(reverse_dict_model)
    summary_df["fs_method"] = summary_df["fs_method"].map(reverse_dict_fs)
    summary_df["label"] = summary_df["label"].map(reverse_dict_label)
    summary_df["data_type"] = summary_df["data_type"].map(reverse_dict_data_type)


    summary_df = summary_df.rename(columns={
        "model": "Model",
        "fs_method": "Feature Selection Method",
        "label": "Prediction Task",
        "num_features": "Number of Features",
        "val_balanced_accuracy": "Validation Balanced Accuracy",
        "test2_balanced_accuracy": "Test Set 2 Balanced Accuracy",
        "test3_balanced_accuracy": "Test Set 3 Balanced Accuracy",
        "val_macro_f1": "Validation Macro F1",
        "test2_macro_f1": "Test Set 2 Macro F1",
        "test3_macro_f1": "Test Set 3 Macro F1",
        "data_type": "Data Type",

    })

    t2 = pd.to_numeric(summary_df["Test Set 2 Balanced Accuracy"], errors="coerce")
    t3 = pd.to_numeric(summary_df["Test Set 3 Balanced Accuracy"], errors="coerce")
    summary_df["Avg External Accuracy"] = ((t2 + t3) / 2).map(smart_round)

    summary_df.sort_values(by=["Validation Macro F1"], ascending=False, inplace=True)

    summary_df.to_csv("model_single_summary_results.csv",index=False)

    print("\n\nNot found folders:")
    for folder in not_found:
        print(folder)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_results()

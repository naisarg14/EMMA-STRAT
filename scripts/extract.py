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
#   Downloads UCEC multi-omics data from LinkedOmics and GDC for four
#   cohort sets, queries GDC file manifests, downloads raw omics files,
#   and merges clinical annotations with molecular subtype and MSI labels.

# ── Imports ───────────────────────────────────────────────────────────────────
import requests
import pandas as pd
import json
from io import StringIO
from pathlib import Path
from tqdm import tqdm
import re
import os
from glob import glob


# ── Functions ─────────────────────────────────────────────────────────────────

def metadata_from_linkedomics():
    #Set 1: TCGA-UCEC
    url = "https://linkedomics.org/data_download/TCGA-UCEC/Human__TCGA_UCEC__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi"

    out_path = Path("metadata") / "set1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    df = pd.read_csv(out_path, sep="\t")
    df = df.set_index(df.columns[0]).T.reset_index()
    df["index"] = df["index"].str.replace(".", "-", regex=False)
    df.rename(columns={"index": "Case_ID"}, inplace=True)
    df.to_csv(out_path, index=False)

    #Set 2: CPTAC-UCEC
    url = "https://linkedomics.org/data_download/CPTAC-UCEC/HS_CPTAC_UCEC_CLI.txt"
    out_path = Path("metadata") / "set2.csv"

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    try:
        df = pd.read_csv(out_path, sep="\t")
    except UnicodeDecodeError:
        df = pd.read_csv(out_path, sep="\t", encoding="cp1252")
    df.rename(columns={"Proteomics_Participant_ID": "Case_ID"}, inplace=True)
    df.to_csv(out_path, index=False)

    #Set 3: CPTAC-UCEC-independent
    url = "https://linkedomics.org/data_download/CPTAC-UCEC-independent/UCEC_confirmatory_meta_table_v3.0.xlsx"
    out_path = Path("metadata") / "set3.csv"

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    df = pd.read_excel(out_path)
    df.rename(columns={"Idx": "Case_ID"}, inplace=True)
    df.to_csv(out_path, index=False)

    #Set 4: CPTAC-pan-cancer-UCEC
    url = "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/UCEC/UCEC_meta.txt"
    out_path = Path("metadata") / "set4.csv"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    df = pd.read_csv(out_path, sep="\t")
    # Set 4 includes a non-relevant second row; remove it.
    if len(df) > 1:
        df = df.drop(df.index[1]).reset_index(drop=True)
    df.rename(columns={"case_id": "Case_ID"}, inplace=True)
    df.to_csv(out_path, index=False)




def query_gdc_files():
    for file in ["set1.csv", "set2.csv", "set3.csv", "set4.csv"]:
        out_path = Path("metadata") / f"gdc_{file}"
        df = pd.read_csv(Path("metadata") / file)
        case_ids = df["Case_ID"].tolist()
        query_gdc(case_ids, out_path)

def query_gdc(case_ids, output_file):
    GDC_SEARCH_URL = "https://api.gdc.cancer.gov/files"

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": case_ids
                }
            },
            {
                "op": "or",
                "content": [
                    {
                        "op": "and",
                        "content": [
                            {
                                "op": "=",
                                "content": {
                                    "field": "files.data_category",
                                    "value": "DNA Methylation"
                                }
                            },
                            {
                                "op": "=",
                                "content": {
                                    "field": "files.data_type",
                                    "value": "Methylation Beta Value"
                                }
                            }
                        ]
                    },
                    # Transcriptome → RNA-seq + miRNA
                    {
                        "op": "and",
                        "content": [
                            {
                                "op": "=",
                                "content": {
                                    "field": "files.data_category",
                                    "value": "Transcriptome Profiling"
                                }
                            },
                            {
                                "op": "in",
                                "content": {
                                    "field": "files.data_type",
                                    "value": [
                                        "Gene Expression Quantification",
                                        "miRNA Expression Quantification"
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }

    payload = {
        "filters": filters,
        "fields": ",".join([
            "file_id",
            "file_name",
            "data_category",
            "data_type",
            "analysis.workflow_type",
            "cases.submitter_id",
            "cases.samples.sample_type",
            "cases.samples.tissue_type",
            "cases.samples.submitter_id"
        ]),
        "format": "TSV",
        "size": 5000
    }

    response = requests.post(GDC_SEARCH_URL, json=payload)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), sep="\t")

    print("Retrieved rows:", len(df))
    print(df.head())

    df.to_csv(output_file, index=False)



def compare_ids():
    files = ["gdc_set1.csv", "gdc_set2.csv", "gdc_set3.csv"]
    id_col = "cases.0.samples.0.submitter_id"

    #print overlap matrix using pandas df
    overlap_matrix = pd.DataFrame(index=files, columns=files)
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i <= j:
                ids1 = pd.read_csv(Path("metadata") / file1)[id_col].tolist()
                ids2 = pd.read_csv(Path("metadata") / file2)[id_col].tolist()
                overlap = len(set(ids1) & set(ids2))
                overlap_matrix.loc[file1, file2] = overlap
                overlap_matrix.loc[file2, file1] = overlap
    print("Overlap matrix:")
    print(overlap_matrix)


def label_sets():
    summary_label = pd.DataFrame()
    summary_msi = pd.DataFrame()
    summary_subtype = pd.DataFrame()

    def update_summary(summary_df: pd.DataFrame, counts: pd.Series, set_name: str) -> pd.DataFrame:
        counts = counts[counts.index.notna()]
        all_columns = summary_df.columns.union(counts.index)
        summary_df = summary_df.reindex(columns=all_columns, fill_value=0)
        if set_name not in summary_df.index:
            summary_df.loc[set_name] = 0
        summary_df.loc[set_name, counts.index] = counts.values
        summary_df = summary_df.sort_index(axis=0).sort_index(axis=1)
        return summary_df.astype(int)


    # for set1 data_clinical_patients.csv has PATIENT_ID,SUBTYPE ; set1.csv has CaseID MSI_phenotype ; gdc_set1.csv has cases.0.samples.0.sample_type, cases.0.samples.0.submitter_id, cases.0.samples.0.tissue_type, cases.0.submitter_id; combine based on PATIENT_ID, CaseID and  cases.0.submitter_id
    df1 = pd.read_csv(Path("metadata") / "set1.csv")
    df2 = pd.read_csv(Path("metadata") / "gdc_set1.csv")
    df3 = pd.read_csv(Path("metadata") / "data_clinical_patient.csv")

    df1 = df1[["Case_ID", "MSI_phenotype"]]
    df2 = df2[["cases.0.samples.0.sample_type", "cases.0.samples.0.submitter_id", "cases.0.samples.0.tissue_type", "cases.0.submitter_id"]]

    merged = pd.merge(df2, df1, right_on="Case_ID", left_on="cases.0.submitter_id", how="outer")
    merged = pd.merge(merged, df3, left_on="Case_ID", right_on="PATIENT_ID", how="inner")

    merged.drop(columns=["PATIENT_ID", "Case_ID", "cases.0.submitter_id", "cases.0.samples.0.sample_type"], inplace=True)
    merged.rename(columns={"MSI_phenotype": "MSI_Status", "SUBTYPE": "Genomic_Subtype", "cases.0.samples.0.submitter_id": "Sample_ID", "cases.0.samples.0.tissue_type": "Tissue_Type"}, inplace=True)

    merged = merged.drop_duplicates(subset=["Sample_ID"])

    merged["Genomic_Subtype"] = merged["Genomic_Subtype"].str.replace("UCEC_", "", regex=False)
    merged.sort_values(by="Tissue_Type", inplace=True, ascending=False)

    normal_mask = merged["Tissue_Type"].eq("Normal")
    merged.loc[normal_mask, ["MSI_Status", "Genomic_Subtype"]] = pd.NA

    merged["Genomic_Subtype"] = merged["Genomic_Subtype"].replace({"CN_LOW": "CNV_LOW", "CN_HIGH": "CNV_HIGH"})

    merged.to_csv(Path("metadata") / "set1_labeled.csv", index=False)

    merged = merged.where(merged.notna(), "Not Labeled")

    tissue_counts = merged["Tissue_Type"].value_counts()
    msi_counts = merged["MSI_Status"].value_counts()
    genomic_counts = merged["Genomic_Subtype"].value_counts()

    summary_label = update_summary(summary_label, tissue_counts, "Set-1: TCGA")
    summary_msi = update_summary(summary_msi, msi_counts, "Set-1: TCGA")
    summary_subtype = update_summary(summary_subtype, genomic_counts, "Set-1: TCGA")



    #for set-2 set2.csv has Genomics_subtype, MSI_status, Case_ID and gdc_set2.csv has cases.0.samples.0.sample_type, cases.0.samples.0.submitter_id, cases.0.samples.0.tissue_type, cases.0.submitter_id; combine based on CaseID and  cases.0.submitter_id
    df1 = pd.read_csv(Path("metadata") / "set2.csv")
    df2 = pd.read_csv(Path("metadata") / "gdc_set2.csv")

    df1 = df1[["Case_ID", "Genomics_subtype", "MSI_status"]]
    df2 = df2[["cases.0.samples.0.sample_type", "cases.0.samples.0.submitter_id", "cases.0.samples.0.tissue_type", "cases.0.submitter_id"]]

    merged = pd.merge(df2, df1, right_on="Case_ID", left_on="cases.0.submitter_id", how="outer")

    merged.drop(columns=["Case_ID", "cases.0.submitter_id", "cases.0.samples.0.sample_type"], inplace=True)
    merged.rename(columns={"Genomics_subtype": "Genomic_Subtype", "MSI_status": "MSI_Status", "cases.0.samples.0.submitter_id": "Sample_ID", "cases.0.samples.0.tissue_type": "Tissue_Type"}, inplace=True)

    merged = merged.drop_duplicates(subset=["Sample_ID"]).dropna(subset=["Sample_ID"])

    merged.sort_values(by="Tissue_Type", inplace=True, ascending=False)

    normal_mask = merged["Tissue_Type"].eq("Normal")
    merged.loc[normal_mask, ["MSI_Status", "Genomic_Subtype"]] = pd.NA

    merged["Genomic_Subtype"] = merged["Genomic_Subtype"].replace({"CNV_low": "CNV_LOW", "CNV_high": "CNV_HIGH", "MSI-H": "MSI"})

    merged.to_csv(Path("metadata") / "set2_labeled.csv", index=False)

    merged = merged.where(merged.notna(), "Not Labeled")

    tissue_counts = merged["Tissue_Type"].value_counts()
    msi_counts = merged["MSI_Status"].value_counts()
    genomic_counts = merged["Genomic_Subtype"].value_counts()
    summary_label = update_summary(summary_label, tissue_counts, "Set-2: CPTAC")
    summary_msi = update_summary(summary_msi, msi_counts, "Set-2: CPTAC")
    summary_subtype = update_summary(summary_subtype, genomic_counts, "Set-2: CPTAC")


    #set 3 has MSI_status,Genomic_subtype, Case_ID and gdc_set3.csv has cases.0.samples.0.sample_type, cases.0.samples.0.submitter_id, cases.0.samples.0.tissue_type, cases.0.submitter_id; combine based on CaseID and  cases.0.submitter_id
    df1 = pd.read_csv(Path("metadata") / "set3.csv")
    df2 = pd.read_csv(Path("metadata") / "gdc_set3.csv")

    df1 = df1[["Case_ID", "Genomic_subtype", "MSI_status"]]
    df2 = df2[["cases.0.samples.0.sample_type", "cases.0.samples.0.submitter_id", "cases.0.samples.0.tissue_type", "cases.0.submitter_id"]]

    merged = pd.merge(df2, df1, right_on="Case_ID", left_on="cases.0.submitter_id", how="outer")

    merged.drop(columns=["Case_ID", "cases.0.submitter_id", "cases.0.samples.0.sample_type"], inplace=True)
    merged.rename(columns={"Genomic_subtype": "Genomic_Subtype", "MSI_status": "MSI_Status", "cases.0.samples.0.submitter_id": "Sample_ID", "cases.0.samples.0.tissue_type": "Tissue_Type"}, inplace=True)
    merged = merged.drop_duplicates(subset=["Sample_ID"]).dropna(subset=["Sample_ID"])

    merged.sort_values(by="Tissue_Type", inplace=True, ascending=False)

    normal_mask = merged["Tissue_Type"].eq("Normal")
    merged.loc[normal_mask, ["MSI_Status", "Genomic_Subtype"]] = pd.NA

    merged["Genomic_Subtype"] = merged["Genomic_Subtype"].replace({"CNV_L": "CNV_LOW", "CNV_H": "CNV_HIGH", "MSI-H": "MSI"})

    merged.to_csv(Path("metadata") / "set3_labeled.csv", index=False)

    merged = merged.where(merged.notna(), "Not Labeled")

    tissue_counts = merged["Tissue_Type"].value_counts()
    msi_counts = merged["MSI_Status"].value_counts()
    genomic_counts = merged["Genomic_Subtype"].value_counts()

    summary_label = update_summary(summary_label, tissue_counts, "Set-3: CPTAC-independent")
    summary_msi = update_summary(summary_msi, msi_counts, "Set-3: CPTAC-independent")
    summary_subtype = update_summary(summary_subtype, genomic_counts, "Set-3: CPTAC-independent")

    print("Summary Tissue Types:")
    print(summary_label)

    print("Summary MSI Status:")
    print(summary_msi)

    print("Summary Genomic Subtypes:")
    print(summary_subtype)

def download_file(file_id, folder="data"):
    try:
        data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"

        response = requests.get(
            data_endpt,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()

        content_disposition = response.headers.get("Content-Disposition", "")
        file_name = re.findall(r'filename=(.+)', content_disposition)[0]

        with open(Path(folder) / file_name, "wb") as f:
            f.write(response.content)

        return file_name
    except Exception as e:
        print(f"Error downloading file with ID {file_id}: {e}")
        return False


def download_files():
    data_types = {
        "Gene Expression Quantification": "rna",
        "miRNA Expression Quantification": "mirna",
        "Methylation Beta Value": "methylation",
    }

    for set_number in [1, 2, 3]:
        df = pd.read_csv(Path("metadata") / f"gdc_set{set_number}.csv")

        downloaded_files = [
                os.path.basename(f)
                for folder in glob(f"data/set{set_number}_*")
                for f in glob(os.path.join(folder, "*"))
                if os.path.isfile(f)
            ]

        for row in tqdm(df.itertuples(), total=len(df), desc=f"Downloading Set {set_number} files"):
            file_id = row.file_id
            data_type = row.data_type
            file_name = row.file_name

            if file_name in downloaded_files:
                continue

            folder_name = f"data/set{set_number}_{data_types.get(data_type, 'other')}"

            Path(folder_name).mkdir(parents=True, exist_ok=True)
            download_file(file_id, folder=folder_name)



def rename_downloaded_files():
    folder_list = ["set1", "set2", "set3"]
    file_types = ["mirna", "rna", "methylation"]

    for folder_name in folder_list:
        print(f"Renaming files in folder: {folder_name}")
        id_df = pd.read_csv(Path("metadata") / f"gdc_{folder_name}.csv")
        id_df = id_df[["file_name", "cases.0.samples.0.submitter_id", "data_type"]]
        for file_type in file_types:
            files = list(Path(f"./data/{folder_name}_{file_type}").glob("*"))
            if "MANIFEST.txt" in [f.name for f in files]:
                files.remove(Path(f"./data/{folder_name}_{file_type}/MANIFEST.txt"))

            for file in tqdm(files):
                matching_row = id_df[id_df["file_name"] == file.name]
                if not matching_row.empty:
                    sample_id = matching_row["cases.0.samples.0.submitter_id"].values[0]
                    data_type = matching_row["data_type"].values[0]
                    if data_type == "miRNA Expression Quantification":
                        new_name = f"{sample_id}_mirna.txt"
                    elif data_type == "Gene Expression Quantification":
                        new_name = f"{sample_id}_rna.txt"
                    elif data_type == "Methylation Beta Value":
                        new_name = f"{sample_id}_methylation.txt"
                    else:
                        continue
                    new_path = file.parent / new_name
                    file.rename(new_path)



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    #metadata_from_linkedomics()
    #query_gdc_files()
    #download_files()
    #compare_ids()
    label_sets()


if __name__ == "__main__":
    main()

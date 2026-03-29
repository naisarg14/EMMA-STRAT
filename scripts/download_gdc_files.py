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
#   Downloads multi-omic files (methylation beta values, RNA-seq, miRNA) from
#   the GDC API using file IDs obtained from gdc_query_results.csv. Each file
#   is parsed and incrementally merged into per-modality CSV outputs to conserve RAM.

# ── Imports ───────────────────────────────────────────────────────────────────
import requests
import re
import os
import pandas as pd
from tqdm import tqdm


# ── Functions ─────────────────────────────────────────────────────────────────
def download_file(file_id):
    try:
        data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"

        response = requests.get(
            data_endpt,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()

        content_disposition = response.headers.get("Content-Disposition", "")
        file_name = re.findall(r'filename=(.+)', content_disposition)[0]

        with open(file_name, "wb") as f:
            f.write(response.content)

        return file_name
    except Exception as e:
        print(f"Error downloading file with ID {file_id}: {e}")
        return False

def process_rnaseq(file_path, sample_name, column_name="tpm_unstranded"):
    df = pd.read_csv(file_path, sep="\t", header=0, skiprows=1)

    unwanted_ids = ['N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous']
    df = df[~df['gene_id'].isin(unwanted_ids)]

    df = df[['gene_id', column_name]]
    df = df.rename(columns={column_name: sample_name})

    return df

def process_mirna(file_path, sample_name, column_name="read_count"):
    df = pd.read_csv(file_path, sep="\t", header=0)
    df = df[['miRNA_ID', column_name]]
    df = df.rename(columns={column_name: sample_name})

    return df

def process_methylation(file_path, sample_name):
    df = pd.read_csv(file_path, sep="\t", header=0)
    if df.shape[1] != 2:
        raise ValueError("Expected exactly 2 columns (CpG_Site, beta value)")

    df.columns = ['CpG_Site', sample_name]
    return df


def merge_and_save(df, id_column, output_file):
    """Merge new data with existing CSV file to save RAM"""
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, low_memory=False)
        merged_df = existing_df.merge(df, on=id_column, how='outer')
        merged_df.to_csv(output_file, index=False)
        del existing_df, merged_df  # Explicitly free memory
    else:
        df.to_csv(output_file, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    download_list = pd.read_csv("gdc_query_results.csv")

    download_list = download_list[download_list['data_type'] == 'Methylation Beta Value']

    for index, row in tqdm(download_list.iterrows(), total=download_list.shape[0]):
        file_id = row['file_id']
        file_name = download_file(file_id)
        download_list.at[index, 'file_name'] = file_name

    for _, row in tqdm(download_list.iterrows(), total=download_list.shape[0]):
        try:
            file_id = row['file_id']
            file_name = row['file_name']
            if not file_name:
                continue

            sample_name = row['cases.0.samples.0.submitter_id']
            data_type = row['data_type']
            cancer_state = row['cases.0.samples.0.tissue_type']

            if data_type == 'Gene Expression Quantification':
                df = process_rnaseq(file_name, sample_name)
                label_row = pd.DataFrame({'gene_id': ['label'], sample_name: [cancer_state]})
                df = pd.concat([df, label_row], ignore_index=True)
                merge_and_save(df, 'gene_id', 'mRNA_Data.csv')
                os.remove(file_name)
            elif data_type == 'miRNA Expression Quantification':
                df = process_mirna(file_name, sample_name)
                label_row = pd.DataFrame({'miRNA_ID': ['label'], sample_name: [cancer_state]})
                df = pd.concat([df, label_row], ignore_index=True)
                merge_and_save(df, 'miRNA_ID', 'miRNA_Data.csv')
                os.remove(file_name)
            elif data_type == 'Methylation Beta Value':
                df = process_methylation(file_name, sample_name)
                label_row = pd.DataFrame({'CpG_Site': ['label'], sample_name: [cancer_state]})
                df = pd.concat([df, label_row], ignore_index=True)
                merge_and_save(df, 'CpG_Site', 'Methylation_Data.csv')
                os.remove(file_name)

            del df
        except Exception as e:
            print(f"Error processing file {file_id}: {e}")
            continue


if __name__ == "__main__":
    main()

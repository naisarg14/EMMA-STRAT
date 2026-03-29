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
#   Extracts TPM values from GDC RNA-seq quantification files, assembles
#   sample-by-gene matrices, and applies low-expression filtering followed
#   by log2(TPM+1) transformation before saving processed output.

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np

# ── Functions ─────────────────────────────────────────────────────────────────

def mrna_file(file):
    df = pd.read_csv(file, sep="\t", header=0, skiprows=1)

    unwanted_ids = ['N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous']
    df = df[~df['gene_id'].isin(unwanted_ids)]

    df.drop(columns=['gene_name', 'gene_type', 'unstranded', 'stranded_first', 'stranded_second', 'fpkm_unstranded', 'fpkm_uq_unstranded'], inplace=True, errors='ignore')

    df.rename(columns={'gene_id': 'Gene_ID', 'tpm_unstranded': 'TPM'}, inplace=True)

    return df

def rna_matrix():
    for set in ['C']:
        folder = f"data/set{set}_rna/"
        files = glob.glob(folder + "*.txt")

        df = None
        for file in tqdm(files):
            sample_id = file.split("/")[-1].replace("_rna.txt", "")
            sample_df = mrna_file(file)
            sample_df.rename(columns={'TPM': sample_id}, inplace=True)

            if df is None:
                df = sample_df
            else:
                df = pd.merge(df, sample_df, on='Gene_ID', how='outer')

        df.fillna(0, inplace=True)
        df = df.set_index('Gene_ID').T
        df.index.name = "Sample_ID"

        print(f"Set {set} shape: {df.shape}")

        df.to_csv(f"data/set{set}_rna_raw.csv")


def preprocess():
    for set in ['B']:
        df = pd.read_csv(f"data/set{set}_rna_raw.csv", index_col=0)
        print(f"Set {set} original shape: {df.shape}")
        filter_threshold = 0.05 * df.shape[0]
        keep_mask = (df >= 1).sum(axis=0) > filter_threshold
        filtered_df = df.loc[:, keep_mask]
        print(f"Set {set} filtered shape: {filtered_df.shape}")

        log2_df = np.log2(filtered_df + 1)

        label_df = pd.read_csv(f"metadata/set{set}_labeled.csv", index_col=0)

        final_df = pd.concat([label_df, log2_df], axis=1, join='inner')

        final_df.to_csv(f"data/set{set}_rna_processed.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rna_matrix()
    #preprocess()

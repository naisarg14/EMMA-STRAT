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
#   Builds per-sample miRNA read-count matrices from GDC quantification files,
#   then normalises to counts-per-million (CPM) and applies log2(CPM+1)
#   transformation with low-expression filtering before saving processed output.

# ── Imports ───────────────────────────────────────────────────────────────────
import glob
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Functions ─────────────────────────────────────────────────────────────────

def mirna_file(file):
    def clean_id(id):
        id_parts = id.split("-")
        if len(id_parts) == 3:
            return id
        elif len(id_parts) == 4:
            return "-".join(id_parts[:3])

    #miRNA_ID	read_count	reads_per_million_miRNA_mapped	cross-mapped
    df = pd.read_csv(file, sep="\t")
    df.drop(columns=['reads_per_million_miRNA_mapped'], inplace=True, errors='ignore')
    #df = df[df['cross-mapped'] == "N"].copy()
    df.drop(columns=['cross-mapped'], inplace=True, errors='ignore')

    df['miRNA_ID'] = df['miRNA_ID'].apply(clean_id)
    # Sum read counts across loci
    collapsed = (
        df.groupby('miRNA_ID', as_index=False)['read_count']
        .sum()
    )

    return collapsed


def mirna_matrix():
    for set in ['C']:
        folder = f"data/set{set}_mirna/"
        files = glob.glob(folder + "*.txt")

        df = None
        for file in tqdm(files):
            sample_id = file.split("/")[-1].replace("_mirna.txt", "")
            sample_df = mirna_file(file)
            sample_df.rename(columns={'read_count': sample_id}, inplace=True)

            if df is None:
                df = sample_df
            else:
                df = pd.merge(df, sample_df, on='miRNA_ID', how='outer')

        df.fillna(0, inplace=True)
        df = df.set_index('miRNA_ID').T
        df.index.name = "Sample_ID"

        print(f"Set {set} shape: {df.shape}")

        df.to_csv(f"data/set{set}_mirna_raw.csv")


def preprocess():
    for set in ['B']:
        df = pd.read_csv(f"data/set{set}_mirna_raw.csv", index_col=0)

        row_sums = df.sum(axis=1)
        row_sums_safe = row_sums.replace(0, 1)
        cpm_df = df.div(row_sums_safe, axis=0) * 1e6
        log2_cpm_df = np.log2(cpm_df + 1)
        print(f"Set {set} CPM shape: {cpm_df.shape}")
        # remove mirna with cpm < 1 in more than 95% of samples
        filter_threshold = math.ceil(0.05 * cpm_df.shape[0])
        keep_mask = (cpm_df >= 1).sum(axis=0) > filter_threshold
        filtered_df = log2_cpm_df.loc[:, keep_mask]
        print(f"Set {set} filtered shape: {filtered_df.shape}")

        label_df = pd.read_csv(f"metadata/set{set}_labeled.csv", index_col=0)

        final_df = pd.concat([label_df, filtered_df], axis=1, join='inner')

        final_df.to_csv(f"data/set{set}_mirna_processed.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mirna_matrix()
    #preprocess()

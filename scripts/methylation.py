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
#   Processes raw methylation beta-value matrices across cohort sets by
#   applying variance filtering, removing high-NA probes, converting beta
#   values to M-values via logit transformation, and attaching sample labels.

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Main ──────────────────────────────────────────────────────────────────────

for set in ['1', '2', '3']:
    df = pd.read_csv(f"data/set{set}_methylation_raw.csv", index_col=0)

    print(f"Original shape: {df.shape}")
    variance = df.var(axis=0)
    #remove variance < 0.01
    filtered_df = df.loc[:, variance >= 0.01].copy()
    print(f"Filtered shape: {filtered_df.shape}")

    #removes rows with >95% NA values
    threshold = int(0.95 * filtered_df.shape[1])
    filtered_df = filtered_df.dropna(thresh=threshold, axis=0)

    print(f"After dropping NA shape: {filtered_df.shape}")

    #convert to M values
    beta_values = filtered_df.clip(lower=1e-6, upper=1-1e-6)
    m_values = pd.DataFrame(
        data = (beta_values / (1 - beta_values)).map(lambda x: np.log2(x)),
        index = filtered_df.index,
        columns = filtered_df.columns
    )

    label_df = pd.read_csv(f"metadata/set{set}_labeled.csv", index_col=0)

    final_df = pd.concat([label_df, m_values], axis=1, join='inner')
    final_df.to_csv(f"data/set{set}_methylation_processed.csv")

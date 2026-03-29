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
#   Prepares and exports transposed multi-omics data (RNA, miRNA, methylation)
#   and clinical labels in the format expected by the Flexynesis benchmarking
#   framework, with a stratified 70/30 split for Set-1.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(19)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = "data_extraction/data"
METADATA_DIR = "data_extraction/metadata"
OUTPUT_DIR = "data_extra"

CLINICAL_COLUMNS = ['Genomic_Subtype', 'MSI_Status', 'Tissue_Type']


# ── Functions ─────────────────────────────────────────────────────────────────
def load_set(set_name: str):
    """Load and align omics + clinical data for one set. Returns filtered DataFrames."""
    rna = pd.read_csv(os.path.join(DATA_DIR, f"{set_name}_rna_model.csv"), index_col=0)
    mirna = pd.read_csv(os.path.join(DATA_DIR, f"{set_name}_mirna_model.csv"), index_col=0)
    meth = pd.read_csv(os.path.join(DATA_DIR, f"{set_name}_methylation_model.csv"), index_col=0)
    clin = pd.read_csv(os.path.join(METADATA_DIR, f"{set_name}_labeled.csv"), index_col=0)

    print(f"[{set_name}] RNA={rna.shape}  miRNA={mirna.shape}  "
          f"Methylation={meth.shape}  Clinical={clin.shape}")

    common = sorted(set(rna.index) & set(mirna.index) & set(meth.index) & set(clin.index))
    print(f"[{set_name}] Common samples: {len(common)}")

    rna = rna.loc[common].copy()
    mirna = mirna.loc[common].copy()
    meth = meth.loc[common].copy()
    clin = clin.loc[common].copy()

    # Drop any clinical label columns that leaked into omics
    for df in [rna, mirna, meth]:
        df.drop(columns=[c for c in CLINICAL_COLUMNS if c in df.columns], inplace=True)

    return common, rna, mirna, meth, clin


def save_split(samples, rna, mirna, meth, clin, out_dir: str, label: str):
    """Transpose omics, subset to samples, and write all four files."""
    os.makedirs(out_dir, exist_ok=True)
    rna.loc[samples].T.to_csv(os.path.join(out_dir, "rna.csv"))
    mirna.loc[samples].T.to_csv(os.path.join(out_dir, "mirna.csv"))
    meth.loc[samples].T.to_csv(os.path.join(out_dir, "methylation.csv"))
    clin.loc[samples].to_csv(os.path.join(out_dir, "clin.csv"))
    print(f"  [{label}] {len(samples)} samples -> {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------
# set-1: 70/30 stratified split
# -----------------------------------------------------------------------
print("=" * 60)
print("SET-1  (train / test split)")
print("=" * 60)

common1, rna1, mirna1, meth1, clin1 = load_set("set1")

stratify = clin1["MSI_Status"].fillna("Unknown")
train_samples, test_samples = train_test_split(
    common1, test_size=0.30, stratify=stratify, random_state=19
)

save_split(train_samples, rna1, mirna1, meth1, clin1,
           os.path.join(OUTPUT_DIR, "train"), "train")
save_split(test_samples, rna1, mirna1, meth1, clin1,
           os.path.join(OUTPUT_DIR, "test"), "test")

print("\nSet-1 clinical distribution:")
for col in clin1.columns:
    print(f"  {col}:")
    print(f"    train: {clin1[col].loc[train_samples].value_counts().to_dict()}")
    print(f"    test:  {clin1[col].loc[test_samples].value_counts().to_dict()}")

# -----------------------------------------------------------------------
# set-2 and set-3: all samples, no split
# -----------------------------------------------------------------------
for set_name in ["set2", "set3"]:
    print()
    print("=" * 60)
    print(f"{set_name.upper()}  (all samples, no split)")
    print("=" * 60)

    common, rna, mirna, meth, clin = load_set(set_name)
    out_dir = os.path.join(OUTPUT_DIR, set_name)
    save_split(common, rna, mirna, meth, clin, out_dir, set_name)

    print(f"\n{set_name} clinical distribution:")
    for col in clin.columns:
        print(f"  {col}: {clin[col].value_counts().to_dict()}")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Done. Output structure:")
print("=" * 60)
print(f"{OUTPUT_DIR}/")
print(f"├── train/   (rna, mirna, methylation, clin) — set-1 70%")
print(f"├── test/    (rna, mirna, methylation, clin) — set-1 30%")
print(f"├── set2/    (rna, mirna, methylation, clin) — all set-2")
print(f"└── set3/    (rna, mirna, methylation, clin) — all set-3")

import pandas as pd


def check_common_features_and_samples():
    for data in ["mirna", "rna", "methylation"]:
        print(f"\nChecking common features for {data}...")
        
        set1 = pd.read_csv(f"data/set1_{data}_processed.csv", index_col=0)
        set2 = pd.read_csv(f"data/set2_{data}_processed.csv", index_col=0)
        set3 = pd.read_csv(f"data/set3_{data}_processed.csv", index_col=0)

        # Find common columns across all three sets
        common_cols = set(set1.columns)
        common_cols.intersection_update(set(set2.columns))
        common_cols.intersection_update(set(set3.columns))

        # Keep these columns and fill NA with 'NA'
        keep_cols = ['Genomic_Subtype', 'MSI_Status', 'Tissue_Type']
        for col in keep_cols:
            set1[col] = set1[col].fillna('NA')
            set2[col] = set2[col].fillna('NA')
            set3[col] = set3[col].fillna('NA')
        
        # Remove columns with NA values except the ones we want to keep
        cols_to_check = [c for c in common_cols if c not in keep_cols]
        valid_cols = [c for c in cols_to_check if not (set1[c].isna().any() or set2[c].isna().any() or set3[c].isna().any())]
        common_cols = set(valid_cols + keep_cols)

        # Filter to common columns
        set1 = set1[list(common_cols)]
        set2 = set2[list(common_cols)]
        set3 = set3[list(common_cols)]

        print(f"{data} common features across sets: {len(common_cols)}")

        # Save to temporary variables for sample filtering
        if data == "mirna":
            mirna1, mirna2, mirna3 = set1, set2, set3
        elif data == "rna":
            rna1, rna2, rna3 = set1, set2, set3
        else:  # methylation
            methyl1, methyl2, methyl3 = set1, set2, set3

    # Now check for common samples across data types for each set
    for s, (mirna, rna, methyl) in enumerate([
        (mirna1, rna1, methyl1),
        (mirna2, rna2, methyl2),
        (mirna3, rna3, methyl3)
    ], start=1):
        print(f"\nChecking common samples for set {s}...")
        print(f"Set {s} samples before filtering: mirna={mirna.shape[0]}, rna={rna.shape[0]}, methyl={methyl.shape[0]}")

        # Find common samples across all three data types
        common_samples = set(mirna.index)
        common_samples.intersection_update(set(rna.index))
        common_samples.intersection_update(set(methyl.index))

        # Filter to common samples and sort
        mirna = mirna.loc[list(common_samples)].sort_index()
        rna = rna.loc[list(common_samples)].sort_index()
        methyl = methyl.loc[list(common_samples)].sort_index()

        # Save filtered data
        mirna.to_csv(f"data/set{s}_mirna_common.csv")
        rna.to_csv(f"data/set{s}_rna_common.csv")
        methyl.to_csv(f"data/set{s}_methylation_common.csv")

        print(f"Set {s} common samples across data types: {len(common_samples)}")


if __name__ == "__main__":
    check_common_features_and_samples()
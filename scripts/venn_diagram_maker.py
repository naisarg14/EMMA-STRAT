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
#   Generates Venn diagrams comparing feature overlap between the best
#   IC (Genomic Subtype) and MSI models for each omics modality
#   (methylation, RNA, miRNA) and all features combined.

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

font = FontProperties(family='Times New Roman', style='italic')

# ── Configuration ─────────────────────────────────────────────────────────────
file1 = "SF_RFELASSO_ic_50.csv"
file2 = "SF_RFESVM_msi_20.csv"

# Create output directories
venn_out = "venn_diagrams"
csv_out = "venn_diagrams"
os.makedirs(venn_out, exist_ok=True)
os.makedirs(csv_out, exist_ok=True)

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Extract features and convert to sets, removing NaN values
methyl1 = set(df1["Methylation"].dropna().to_list())
methyl2 = set(df2["Methylation"].dropna().to_list())

rna1 = set(df1["RNA"].dropna().to_list())
rna2 = set(df2["RNA"].dropna().to_list())

mirna1 = set(df1["miRNA"].dropna().to_list())
mirna2 = set(df2["miRNA"].dropna().to_list())

all1 = methyl1 | rna1 | mirna1
all2 = methyl2 | rna2 | mirna2


# ── Functions ─────────────────────────────────────────────────────────────────
def create_venn_diagram(set1, set2, title, filename):
    """Create a clean, professional venn diagram with equal-sized circles"""
    fig, ax = plt.subplots(figsize=(10, 8))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Calculate overlaps
    only_in_1 = len(set1 - set2)
    only_in_2 = len(set2 - set1)
    common = len(set1 & set2)
    
    # Create two equal-sized circles with soft colors
    ##c1d3ba
#cbc0d3
    circle1 = Circle((0.35, 0.5), 0.35, color='#ffbe5c', alpha=0.7, ec='#666666', linewidth=2)
    circle2 = Circle((0.65, 0.5), 0.35, color='#a8e9a8', alpha=0.7, ec='#666666', linewidth=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Add text labels and counts
    # Left circle (IC) - only IC
    ax.text(0.20, 0.5, str(only_in_1), fontsize=18, fontweight='bold', 
            ha='center', va='center', color='#1a1a1a')
    
    # Center - common
    ax.text(0.5, 0.5, str(common), fontsize=18, fontweight='bold', 
            ha='center', va='center', color='#1a1a1a')
    
    # Right circle (MSI) - only MSI  
    ax.text(0.80, 0.5, str(only_in_2), fontsize=18, fontweight='bold', 
            ha='center', va='center', color='#1a1a1a')
    
    # Set labels with total counts
    ax.text(0.2, 0.95, f'Genomic Subtype\n({len(set1)})', fontsize=12, fontweight='bold', 
            ha='center', va='top', color='#333333', family='sans-serif')
    ax.text(0.8, 0.95, f'MSI\n({len(set2)})', fontsize=12, fontweight='bold', 
            ha='center', va='top', color='#333333', family='sans-serif')
    
    # Set axis limits and remove axes
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save with clean white background
    plt.title(title, fontsize=16, fontweight='bold', color='#333333', family='sans-serif')
    plt.savefig(os.path.join(venn_out, filename), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.close()


def create_venn_csv(set1, set2, data_name):
    """Create CSV with features and unions"""
    # Calculate intersections and differences
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    common = set1 & set2
    union = set1 | set2
    
    # Create summary data
    max_len = max(len(only_in_1), len(only_in_2), len(common), len(union))
    
    summary_data = {
        f'{file1.split(".")[0]}_only': list(only_in_1) + [''] * (max_len - len(only_in_1)),
        f'{file2.split(".")[0]}_only': list(only_in_2) + [''] * (max_len - len(only_in_2)),
        'Common': list(common) + [''] * (max_len - len(common)),
        'Union': list(union) + [''] * (max_len - len(union))
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(csv_out, f'{data_name}_venn.csv'), index=False)
    
    # Create detailed statistics CSV
    stats = {
        'Category': [f'{file1.split(".")[0]}_only', f'{file2.split(".")[0]}_only', 'Common', 'Union'],
        'Count': [len(only_in_1), len(only_in_2), len(common), len(union)],
        'Percentage': [
            f'{len(only_in_1)/len(union)*100:.2f}%' if union else 0,
            f'{len(only_in_2)/len(union)*100:.2f}%' if union else 0,
            f'{len(common)/len(union)*100:.2f}%' if union else 0,
            '100%'
        ]
    }
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(csv_out, f'{data_name}_venn_stats.csv'), index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
# Create venn diagrams for each data type
create_venn_diagram(methyl1, methyl2, 'Methylation', 'methyl_venn.png')
create_venn_diagram(rna1, rna2, 'RNA', 'rna_venn.png')
create_venn_diagram(mirna1, mirna2, 'miRNA', 'mirna_venn.png')
create_venn_diagram(all1, all2, 'All Features', 'all_venn.png')

# Create CSV files for each data type
create_venn_csv(methyl1, methyl2, 'methyl')
create_venn_csv(rna1, rna2, 'rna')
create_venn_csv(mirna1, mirna2, 'mirna')
create_venn_csv(all1, all2, 'all')
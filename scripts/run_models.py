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
#   Orchestrates the execution of all ML model training scripts by iterating
#   over every combination of model code, feature count, feature selection
#   method, and prediction label, invoking each as a subprocess.

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import subprocess

# ── Configuration ─────────────────────────────────────────────────────────────
codes = sys.argv[1:]
features = [10, 20, 50, 100, 150, 200]
fs_method = ["RFERF", "RFESVM", "RFELASSO", "ANOVA"]
labels = ["ic", "msi"]
selected_fs = ["mirnas", "rna", "methyl"]

# ── Functions ─────────────────────────────────────────────────────────────────
def run_models():
    count = 1
    total = len(codes) * len(features) * len(fs_method) * len(labels)
    for code in codes:
        for num in features:
            for fs in fs_method:
                for label in labels:
                    print(f"\nRunning {count} out of {total}. Completed: {((count-1)/total)*100:.2f}%")
                    cmd = ["python3", code, label, fs, str(num)]
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    count += 1

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_models()

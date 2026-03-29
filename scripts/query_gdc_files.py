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
#   Queries the GDC (Genomic Data Commons) API to retrieve file IDs and metadata
#   for DNA methylation (beta values), gene expression, and miRNA data associated
#   with TCGA case IDs. Results are saved to a CSV for downstream downloading.

# ── Imports ───────────────────────────────────────────────────────────────────
import requests
import pandas as pd
import json
from io import StringIO

# ── Configuration ─────────────────────────────────────────────────────────────
# Load case IDs
case_id_df = pd.read_csv("cases.tsv", sep="\t")
case_ids = case_id_df["submitter_id"].tolist()

GDC_SEARCH_URL = "https://api.gdc.cancer.gov/files"

# ── Main ──────────────────────────────────────────────────────────────────────
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
                # DNA methylation → Masked Intensities
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

df.to_csv("gdc_query_results.csv", index=False)

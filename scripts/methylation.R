library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylation27kanno.ilmn12.hg19)
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(dplyr)
library(tidyr)
library(tibble)
library(data.table)
library(future.apply)

setwd("C:/N Drive/13. Novel ML/data_extraction/")
cat("Working directory:", getwd(), "\n")

# ── 1. Load & pre-process annotations once ────────────────────────────────────

prepare_anno <- function(anno, gene_col, region_col) {
  anno %>%
    as.data.frame() %>%
    select(probe_id = Name, gene = all_of(gene_col), region = all_of(region_col)) %>%
    separate_rows(gene,   sep = ";") %>%
    separate_rows(region, sep = ";") %>%
    filter(region %in% c("TSS200", "TSS1500"), gene != "") %>%
    distinct(probe_id, gene) %>%
    as.data.table() %>%
    setkey("probe_id")
}

cat("Preparing annotations...\n")
anno_450k_clean <- prepare_anno(
  getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19),
  "UCSC_RefGene_Name", "UCSC_RefGene_Group"
)
anno_epic_clean <- prepare_anno(
  getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19),
  "UCSC_RefGene_Name", "UCSC_RefGene_Group"
)
prepare_anno_27k <- function(anno) {
  anno %>%
    as.data.frame() %>%
    select(
      probe_id        = Name,
      gene            = Symbol,           # clean gene symbol e.g. "ATP2A1"
      distance_to_tss = Distance_to_TSS
    ) %>%
    filter(
      !is.na(distance_to_tss),
      distance_to_tss <= 1500,            # equivalent to TSS200 + TSS1500
      gene != ""
    ) %>%
    distinct(probe_id, gene) %>%
    as.data.table() %>%
    setkey("probe_id")
}

anno_27k_clean <- prepare_anno_27k(
  getAnnotation(IlluminaHumanMethylation27kanno.ilmn12.hg19)
)
cat("Annotations ready.\n")

# ── 2. Platform detection ──────────────────────────────────────────────────────

detect_platform <- function(probes) {
  if (!any(grepl("^cg", probes))) stop("Unknown platform")
  n <- length(probes)
  if (n > 700000) return("epic")
  if (n > 100000) return("450k")
  return("27k")
}

# ── 3. Per-sample processing (data.table throughout) ──────────────────────────

process_sample <- function(file) {
  sample_id <- gsub("\\..*$", "", basename(file))

  betas <- fread(
    file,
    header    = FALSE,
    col.names = c("probe_id", "beta"),
    key       = "probe_id"
  )

  platform <- detect_platform(betas$probe_id)

  anno <- switch(platform,
    "epic" = anno_epic_clean,
    "450k" = anno_450k_clean,
    "27k"  = anno_27k_clean
  )

  df <- anno[betas, on = "probe_id", nomatch = 0][
    , .(beta = mean(beta, na.rm = TRUE), n_cpgs = .N), by = gene
  ][n_cpgs >= 2, .(gene, beta)]

  setnames(df, "beta", sample_id)

  list(data = df, platform = platform, sample = sample_id)
}

# ── 4. Parallel execution ──────────────────────────────────────────────────────

input_dir <- "./data/setB_methylation/"
files     <- list.files(input_dir, pattern = "\\.txt$", full.names = TRUE)

if (length(files) == 0) stop("No .txt files found in ", input_dir)
cat("Processing", length(files), "methylation files...\n")

plan(multisession, workers = max(1, parallel::detectCores() - 1))
results <- future_lapply(files, process_sample, future.seed = TRUE)
plan(sequential)  # release workers

# ── 5. Platform summary ────────────────────────────────────────────────────────

platform_info <- data.frame(
  sample   = sapply(results, function(x) x$sample),
  platform = sapply(results, function(x) x$platform)
)

cat("\n=== Platform Summary ===\n")
print(table(platform_info$platform))
cat("Total samples:", nrow(platform_info), "\n\n")

# ── 6. Merge all samples into one matrix ──────────────────────────────────────

cat("Merging samples...\n")

gene_beta_matrix <- Reduce(
  function(x, y) merge(x, y, by = "gene", all = TRUE),
  lapply(results, function(x) x$data)
)

# ── 7. Transpose: rows = samples, cols = genes ────────────────────────────────

gene_beta_transposed <- transpose(
  gene_beta_matrix,
  keep.names = "Sample_ID",
  make.names = "gene"
)

# Row names after transpose are the original column names (sample IDs)
# The first row after transpose is "gene" label — drop it and fix Sample_ID
gene_beta_transposed <- gene_beta_transposed[-1]  # drop the "gene" header row
gene_beta_transposed[, Sample_ID := gsub("_methylation$", "", Sample_ID)]

# ── 8. Write output ───────────────────────────────────────────────────────────

out_path <- "data/setB_methylation_raw2.csv"
fwrite(gene_beta_transposed, out_path, sep = ",", na = "NA")
cat("Written to", out_path, "\n")
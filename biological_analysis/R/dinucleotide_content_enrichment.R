#!/usr/bin/env Rscript
#
# dinucl_enrichment.R
#
# Compute per-dinucleotide Cliff's delta and AUROC between an experimental
# and a reference set of sequences, given pre-computed dinucleotide frequency
# tables.
#
# The script:
#   1. Reads dinucleotide frequency TSVs for the reference and experimental
#      sets.  The first N numeric columns are treated as dinucleotide
#      frequencies (default: the first 16 columns = 4^2 canonical
#      dinucleotides).
#   2. Optionally reads a labels file (seqid <TAB> sequence <TAB> method)
#      to split the experimental set by method.  When no labels file is
#      provided, all experimental sequences are treated as a single group.
#   3. Computes Cliff's delta for each dinucleotide between the
#      experimental group(s) and the reference set.
#   4. Derives AUROC via:  AUROC = (delta + 1) / 2
#   5. Writes a long-format TSV with one row per (method x dinucleotide).
#
# Usage:
#   Rscript dinucl_enrichment.R \
#     --exp_freq    <experimental_dinucl_freq.tsv> \
#     --ref_freq    <reference_dinucl_freq.tsv> \
#     --output      <results.tsv> \
#     [--exp_labels  <experimental_sequences.tsv>] \
#     [--n_dinucl    16]

suppressPackageStartupMessages({
  library(argparse)
  library(dplyr)
  library(effectsize)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser <- ArgumentParser(
  description = paste(
    "Per-dinucleotide Cliff's delta & AUROC between experimental and",
    "reference sequence sets from pre-computed frequency tables."
  )
)
parser$add_argument("--exp_freq", required = TRUE,
                    help = "Dinucleotide frequency TSV for the experimental sequences (header required).")
parser$add_argument("--ref_freq", required = TRUE,
                    help = "Dinucleotide frequency TSV for the reference / pool sequences (header required).")
parser$add_argument("--output", required = TRUE,
                    help = "Output TSV path (long format: method x dinucleotide).")
parser$add_argument("--exp_labels", default = NULL,
                    help = paste(
                      "Optional 3-column TSV (no header): seqid, sequence, method.",
                      "Rows must align 1-to-1 with --exp_freq.",
                      "When provided, Cliff's delta is computed per method."
                    ))
parser$add_argument("--n_dinucl", type = "integer", default = 16L,
                    help = "Number of leading columns in the frequency files that are dinucleotide frequencies [default: 16].")

args <- parser$parse_args()


# ---------------------------------------------------------------------------
# Load frequency tables
# ---------------------------------------------------------------------------
read_freq <- function(path, n_dinucl) {
  df <- read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                   check.names = FALSE)
  if (ncol(df) < n_dinucl) {
    stop("Expected at least ", n_dinucl, " columns in ", path,
         " but found ", ncol(df), ".")
  }
  df
}


# ---------------------------------------------------------------------------
# Compute Cliff's delta per dinucleotide for one group vs reference
# ---------------------------------------------------------------------------
cliff_delta_per_dinucl <- function(exp_df, ref_df, n_dinucl, method_label) {
  dinucl_names <- colnames(ref_df)[seq_len(n_dinucl)]
  
  results <- lapply(seq_len(n_dinucl), function(i) {
    cd <- cliffs_delta(exp_df[[i]], ref_df[[i]])
    delta <- cd$r_rank_biserial
    data.frame(
      method           = method_label,
      dinucleotide     = dinucl_names[i],
      cliffs_delta     = delta,
      cliffs_delta_CI_low  = cd$CI_low,
      cliffs_delta_CI_high = cd$CI_high,
      auroc            = (delta + 1) / 2,
      n_exp            = nrow(exp_df),
      n_ref            = nrow(ref_df),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, results)
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main <- function() {
  n <- args$n_dinucl
  
  message("Reading reference frequencies: ", args$ref_freq)
  ref_freq <- read_freq(args$ref_freq, n)
  message("  ", nrow(ref_freq), " sequences, ", n, " dinucleotide columns.")
  
  message("Reading experimental frequencies: ", args$exp_freq)
  exp_freq <- read_freq(args$exp_freq, n)
  message("  ", nrow(exp_freq), " sequences.")
  
  # Determine grouping
  if (!is.null(args$exp_labels)) {
    message("Reading labels: ", args$exp_labels)
    labs <- read.table(args$exp_labels, header = FALSE, sep = "\t",
                       col.names = c("seqid", "seq", "method"),
                       stringsAsFactors = FALSE)
    if (nrow(labs) != nrow(exp_freq)) {
      stop("Label file has ", nrow(labs), " rows but frequency file has ",
           nrow(exp_freq), ". They must match 1-to-1.")
    }
    exp_freq$method <- labs$method
    methods <- unique(exp_freq$method)
    message("  ", length(methods), " methods found.")
    
    all_results <- lapply(methods, function(m) {
      subset_df <- exp_freq[exp_freq$method == m, ]
      cliff_delta_per_dinucl(subset_df, ref_freq, n, method_label = m)
    })
    results <- do.call(rbind, all_results)
  } else {
    message("No labels file provided — treating all experimental sequences as one group.")
    results <- cliff_delta_per_dinucl(exp_freq, ref_freq, n, method_label = "experimental")
  }
  
  write.table(results, file = args$output, sep = "\t", row.names = FALSE,
              quote = FALSE)
  message("Results written to: ", args$output)
  message("")
  print(results)
}

main()
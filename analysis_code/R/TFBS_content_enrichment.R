#!/usr/bin/env Rscript
#
# tfbs_enrichment.R
#
# Compute Cliff's delta and AUROC for TF motif count enrichment between an
# experimental set of sequences and a reference (background) set, using FIMO
# output.
#
# The script:
#   1. Reads FIMO TSV output for the reference (AL pool) sequences.
#   2. Reads sequence tables for the reference and experimental conditions.
#   3. Left-joins FIMO hits onto sequences so that sequences with zero motif
#      hits are retained with count = 0.
#   4. Counts total TF motif hits per sequence.
#   5. Computes Cliff's delta (rank-biserial correlation) between the
#      experimental and reference count distributions.
#   6. Derives AUROC from the Mann-Whitney U via:  AUROC = (delta + 1) / 2
#   7. Writes both metrics to a TSV.
#
# Usage:
#   Rscript tfbs_enrichment.R \
#     --fimo       <fimo.tsv or directory of fimo.tsv files> \
#     --ref_seq    <reference_sequences.tsv> \
#     --exp_seq    <experimental_sequences.tsv> \
#     --output     <results.tsv> \
#     [--motif_col       motif_id | motif_alt_id] \
#     [--split_motif_id] \
#     [--tf_whitelist     <file with one TF ID per line>]
#
# Input file formats:
#   fimo.tsv       — Standard FIMO output (tab-separated, with header).
#   *_sequences.tsv — 3-column tab-separated, NO header:
#                      sequence_name <TAB> sequence <TAB> method
#                    sequence_name may optionally be pipe-delimited as
#                    "seqid|method"; the script handles both cases.
#   tf_whitelist   — One TF identifier per line (optional).

suppressPackageStartupMessages({
  library(argparse)
  library(stringr)
  library(dplyr)
  library(effectsize)
})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser <- ArgumentParser(
  description = "TF motif enrichment: Cliff's delta & AUROC from FIMO output"
)
parser$add_argument("--fimo", required = TRUE,
                    help = "Path to a FIMO TSV file, or a directory containing fimo.tsv files (searched recursively).")
parser$add_argument("--ref_seq", required = TRUE,
                    help = "Path to the reference (background / AL pool) sequence table (3-col TSV, no header).")
parser$add_argument("--exp_seq", required = TRUE,
                    help = "Path to the experimental sequence table (3-col TSV, no header).")
parser$add_argument("--output", required = TRUE,
                    help = "Path for the output TSV with Cliff's delta and AUROC per method.")
parser$add_argument("--motif_col", default = "motif_id",
                    help = "Column name in the FIMO file to use as the TF identifier [default: motif_id].")
parser$add_argument("--split_motif_id", action = "store_true", default = FALSE,
                    help = "If set, split motif_id on '.' and keep the first element (for HOCOMOCO IDs).")
parser$add_argument("--tf_whitelist", default = NULL,
                    help = "Optional file with one TF ID per line. Only these TFs will be retained.")

args <- parser$parse_args()


# ---------------------------------------------------------------------------
# Helper: read sequence table
# ---------------------------------------------------------------------------
read_seq_table <- function(path) {
  df <- read.table(path, header = FALSE, sep = "\t",
                   col.names = c("sequence_name", "sequence", "method"),
                   stringsAsFactors = FALSE)
  # If sequence_name contains a pipe, the method is encoded there; parse it.
  has_pipe <- any(grepl("\\|", df$sequence_name))
  if (has_pipe) {
    parts <- str_split_fixed(df$sequence_name, "\\|", 2)
    df$sequence_name <- parts[, 1]
    # Only overwrite method if the pipe-part is non-empty
    pipe_method <- parts[, 2]
    df$method[pipe_method != ""] <- pipe_method[pipe_method != ""]
  }
  df
}


# ---------------------------------------------------------------------------
# Step 1: Load FIMO hits
# ---------------------------------------------------------------------------
load_fimo <- function(path, motif_col, split_motif_id, tf_whitelist) {
  if (dir.exists(path)) {
    files <- list.files(path, pattern = "fimo\\.tsv$", recursive = TRUE,
                        full.names = TRUE)
    if (length(files) == 0) stop("No fimo.tsv files found in directory: ", path)
    fimo <- do.call(rbind, lapply(files, function(f) {
      read.table(f, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                 comment.char = "#")
    }))
  } else {
    fimo <- read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                       comment.char = "#")
  }
  
  # Normalise the motif column to "motif_id"
  if (motif_col != "motif_id") {
    if (!(motif_col %in% colnames(fimo)))
      stop("Column '", motif_col, "' not found in FIMO output.")
    fimo$motif_id <- fimo[[motif_col]]
  }
  
  if (split_motif_id) {
    fimo$motif_id <- str_split_fixed(fimo$motif_id, "\\.", 2)[, 1]
  }
  
  # Parse pipe-delimited sequence_name if present
  if (any(grepl("\\|", fimo$sequence_name))) {
    parts <- str_split_fixed(fimo$sequence_name, "\\|", 2)
    fimo$sequence_name <- parts[, 1]
  }
  
  # Apply whitelist
  if (!is.null(tf_whitelist)) {
    fimo <- fimo[fimo$motif_id %in% tf_whitelist, ]
  }
  
  fimo[, c("sequence_name", "motif_id")]
}


# ---------------------------------------------------------------------------
# Step 2: Merge FIMO hits with a sequence table (left join → zero-hit seqs
#          get motif_id = 0)
# ---------------------------------------------------------------------------
merge_fimo_with_seqs <- function(fimo_hits, seq_df) {
  merged <- merge(
    seq_df[, c("sequence_name", "method")],
    fimo_hits,
    by = "sequence_name",
    all.x = TRUE
  )
  merged$motif_id[is.na(merged$motif_id)] <- 0
  merged
}


# ---------------------------------------------------------------------------
# Step 3: Count total motif hits per sequence
# ---------------------------------------------------------------------------
count_motifs_per_seq <- function(merged_df) {
  has_hit <- merged_df[merged_df$motif_id != 0, ]
  counts_hit <- has_hit %>%
    group_by(sequence_name, method, motif_id) %>%
    summarise(counts = n(), .groups = "drop") %>%
    group_by(sequence_name, method) %>%
    summarise(counts = sum(counts), .groups = "drop")
  
  no_hit <- merged_df[merged_df$motif_id == 0, ]
  if (nrow(no_hit) > 0) {
    counts_zero <- no_hit %>%
      distinct(sequence_name, method) %>%
      mutate(counts = 0L)
    counts_all <- bind_rows(counts_hit, counts_zero)
  } else {
    counts_all <- counts_hit
  }
  
  counts_all
}


# ---------------------------------------------------------------------------
# Step 4: Cliff's delta + AUROC
# ---------------------------------------------------------------------------
compute_enrichment <- function(counts_exp, counts_ref) {
  methods <- unique(counts_exp$method)
  results <- lapply(methods, function(m) {
    x <- counts_exp$counts[counts_exp$method == m]
    y <- counts_ref$counts
    cd <- cliffs_delta(x, y)
    delta <- cd$r_rank_biserial
    ci_lo <- cd$CI_low
    ci_hi <- cd$CI_high
    auroc <- (delta + 1) / 2
    
    data.frame(
      method           = m,
      cliffs_delta     = delta,
      cliffs_delta_CI_low  = ci_lo,
      cliffs_delta_CI_high = ci_hi,
      auroc            = auroc,
      n_exp            = length(x),
      n_ref            = length(y),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, results)
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main <- function() {
  # Optional whitelist
  tf_whitelist <- NULL
  if (!is.null(args$tf_whitelist)) {
    tf_whitelist <- readLines(args$tf_whitelist)
    tf_whitelist <- trimws(tf_whitelist)
    tf_whitelist <- tf_whitelist[tf_whitelist != ""]
    message("Loaded ", length(tf_whitelist), " TF IDs from whitelist.")
  }
  
  # Load FIMO
  message("Loading FIMO results from: ", args$fimo)
  fimo_hits <- load_fimo(args$fimo, args$motif_col,
                         args$split_motif_id, tf_whitelist)
  message("  ", nrow(fimo_hits), " motif hits after filtering.")
  
  # Load sequence tables
  message("Reading reference sequences: ", args$ref_seq)
  ref_seqs <- read_seq_table(args$ref_seq)
  message("  ", nrow(ref_seqs), " sequences.")
  
  message("Reading experimental sequences: ", args$exp_seq)
  exp_seqs <- read_seq_table(args$exp_seq)
  message("  ", nrow(exp_seqs), " sequences.")
  
  # Merge & count — reference
  ref_merged <- merge_fimo_with_seqs(fimo_hits, ref_seqs)
  ref_counts <- count_motifs_per_seq(ref_merged)
  
  # Merge & count — experimental
  exp_merged <- merge_fimo_with_seqs(fimo_hits, exp_seqs)
  exp_counts <- count_motifs_per_seq(exp_merged)
  
  # Compute enrichment per method in the experimental set
  message("Computing Cliff's delta and AUROC...")
  results <- compute_enrichment(exp_counts, ref_counts)
  
  # Write output
  write.table(results, file = args$output, sep = "\t", row.names = FALSE,
              quote = FALSE)
  message("Results written to: ", args$output)
  message("")
  print(results)
}

main()
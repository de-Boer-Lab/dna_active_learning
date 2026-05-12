#!/usr/bin/env Rscript
#
# tf_enrichment_fisher.R
#
# Per-TF enrichment of TFBS between an experimental (foreground) and a
# reference (background) set of sequences, quantified by Fisher's exact test
# on sequence-level presence / absence.
#
# The script:
#   1. Reads FIMO TSV output for the foreground and background conditions.
#      These can be separate files, or the same file when the scan was run
#      on a combined pool.
#   2. Reads sequence-name lists for both sets so that sequences with zero
#      FIMO hits are properly counted in the contingency table.
#   3. Left-joins FIMO hits onto each sequence list (0 for no hit).
#   4. Builds a binary presence/absence matrix per (sequence, TF).
#   5. For every TF observed in either set, constructs a 2×2 contingency
#      table and runs Fisher's exact test (two-sided).  A Haldane–Anscombe
#      correction (+0.5) is applied to the odds ratio when any cell is zero.
#   6. BH-adjusts p-values across all TFs (within each foreground method
#      when a method column is present).
#   7. Writes a TSV with: motif_id, odds_ratio, pvalue, padj, ci_low,
#      ci_high, and supporting counts.
#
# Usage:
#   Rscript tf_enrichment_fisher.R \
#     --fimo_fg      <fimo_foreground.tsv or directory> \
#     --fimo_bg      <fimo_background.tsv or directory> \
#     --seq_fg       <foreground_sequences.tsv> \
#     --seq_bg       <background_sequences.tsv> \
#     --output       <results.tsv> \
#     [--motif_col       motif_id | motif_alt_id] \
#     [--split_motif_id] \
#     [--tf_whitelist     <file with one TF ID per line>]
#
# Input file formats:
#   fimo*.tsv       — Standard FIMO output (tab-separated, with header).
#                     A path to a directory is also accepted; all fimo.tsv
#                     files found recursively will be concatenated.
#   *_sequences.tsv — 3-column tab-separated, NO header:
#                       sequence_name <TAB> sequence <TAB> method
#                     sequence_name may optionally be pipe-delimited as
#                     "seqid|method"; the script handles both cases.
#                     When the foreground file contains multiple methods,
#                     a separate Fisher test is run per method and padj is
#                     computed within each method.
#   tf_whitelist    — One TF identifier per line (optional).

suppressPackageStartupMessages({
  library(argparse)
  library(stringr)
  library(dplyr)
  library(purrr)
})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser <- ArgumentParser(
  description = "Per-TF Fisher's exact test enrichment from FIMO output"
)
parser$add_argument("--fimo_fg", required = TRUE,
                    help = "FIMO TSV (or directory of fimo.tsv files) for foreground sequences.")
parser$add_argument("--fimo_bg", required = TRUE,
                    help = "FIMO TSV (or directory of fimo.tsv files) for background sequences.")
parser$add_argument("--seq_fg", required = TRUE,
                    help = "Foreground sequence table (3-col TSV, no header: seqid, seq, method).")
parser$add_argument("--seq_bg", required = TRUE,
                    help = "Background sequence table (3-col TSV, no header: seqid, seq, method).")
parser$add_argument("--output", required = TRUE,
                    help = "Output TSV path.")
parser$add_argument("--motif_col", default = "motif_id",
                    help = "Column name in FIMO output for the TF identifier [default: motif_id].")
parser$add_argument("--split_motif_id", action = "store_true", default = FALSE,
                    help = "Split motif_id on '.' and keep the first element (for HOCOMOCO IDs).")
parser$add_argument("--tf_whitelist", default = NULL,
                    help = "Optional file with one TF ID per line; only these TFs are tested.")

args <- parser$parse_args()


# ---------------------------------------------------------------------------
# Helper: read sequence table → (sequence_name, method)
# ---------------------------------------------------------------------------
read_seq_table <- function(path) {
  df <- read.table(path, header = FALSE, sep = "\t",
                   col.names = c("sequence_name", "sequence", "method"),
                   stringsAsFactors = FALSE)
  # Handle pipe-delimited sequence_name
  if (any(grepl("\\|", df$sequence_name))) {
    parts <- str_split_fixed(df$sequence_name, "\\|", 2)
    df$sequence_name <- parts[, 1]
    pipe_method <- parts[, 2]
    df$method[pipe_method != ""] <- pipe_method[pipe_method != ""]
  }
  df$sequence <- NULL
  df
}


# ---------------------------------------------------------------------------
# Helper: load FIMO hits → (sequence_name, motif_id)
# ---------------------------------------------------------------------------
load_fimo <- function(path, motif_col, split_motif_id, tf_whitelist) {
  if (dir.exists(path)) {
    files <- list.files(path, pattern = "fimo\\.tsv$", recursive = TRUE,
                        full.names = TRUE)
    if (length(files) == 0) stop("No fimo.tsv files found in: ", path)
    fimo <- do.call(rbind, lapply(files, function(f) {
      read.table(f, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                 comment.char = "#")
    }))
  } else {
    fimo <- read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                       comment.char = "#")
  }
  
  if (motif_col != "motif_id") {
    if (!(motif_col %in% colnames(fimo)))
      stop("Column '", motif_col, "' not found in FIMO output.")
    fimo$motif_id <- fimo[[motif_col]]
  }
  
  if (split_motif_id) {
    fimo$motif_id <- str_split_fixed(fimo$motif_id, "\\.", 2)[, 1]
  }
  
  # Clean pipe-delimited sequence_name
  if (any(grepl("\\|", fimo$sequence_name))) {
    fimo$sequence_name <- str_split_fixed(fimo$sequence_name, "\\|", 2)[, 1]
  }
  
  if (!is.null(tf_whitelist)) {
    fimo <- fimo[fimo$motif_id %in% tf_whitelist, ]
  }
  
  fimo[, c("sequence_name", "motif_id")]
}


# ---------------------------------------------------------------------------
# Helper: merge FIMO hits with sequence list (left join, 0 for no hit)
# ---------------------------------------------------------------------------
merge_fimo_with_seqs <- function(fimo_hits, seq_df) {
  merged <- merge(seq_df, fimo_hits, by = "sequence_name", all.x = TRUE)
  merged$motif_id[is.na(merged$motif_id)] <- "0"
  merged
}


# ---------------------------------------------------------------------------
# Build presence/absence: unique (sequence_name, motif_id) pairs with hit
# ---------------------------------------------------------------------------
make_presence <- function(merged_df) {
  merged_df %>%
    filter(motif_id != "0") %>%
    distinct(sequence_name, motif_id)
}


# ---------------------------------------------------------------------------
# Fisher's exact test for one TF
# ---------------------------------------------------------------------------
fisher_one_tf <- function(motif, fg_seqs, bg_seqs,
                          fg_present_seqs, bg_present_seqs,
                          n_fg, n_bg) {
  a <- sum(fg_seqs %in% fg_present_seqs)
  b <- n_fg - a
  cc <- sum(bg_seqs %in% bg_present_seqs)
  d <- n_bg - cc
  
  # Skip TFs with zero hits in both sets
  if ((a + cc) == 0) return(NULL)
  
  # Haldane-Anscombe correction for OR when any cell is zero
  a_c <- ifelse(a == 0, 0.5, a)
  b_c <- ifelse(b == 0, 0.5, b)
  c_c <- ifelse(cc == 0, 0.5, cc)
  d_c <- ifelse(d == 0, 0.5, d)
  
  mat <- matrix(c(a, b, cc, d), nrow = 2,
                dimnames = list(c("with_motif", "without_motif"),
                                c("foreground", "background")))
  ft <- fisher.test(mat, alternative = "two.sided")
  
  corrected_or <- (a_c / b_c) / (c_c / d_c)
  final_or <- ifelse(any(c(a, b, cc, d) == 0), corrected_or, ft$estimate)
  
  data.frame(
    motif_id   = motif,
    n_fg_hit   = a,
    n_fg_total = n_fg,
    n_bg_hit   = cc,
    n_bg_total = n_bg,
    odds_ratio = final_or,
    pvalue     = ft$p.value,
    ci_low     = ft$conf.int[1],
    ci_high    = ft$conf.int[2],
    zero_cell  = any(c(a, b, cc, d) == 0),
    stringsAsFactors = FALSE
  )
}


# ---------------------------------------------------------------------------
# Run Fisher enrichment for one foreground method
# ---------------------------------------------------------------------------
run_fisher_one_method <- function(fg_merged, bg_presence, bg_seqs, n_bg,
                                  method_label) {
  fg_seqs <- unique(fg_merged$sequence_name)
  n_fg <- length(fg_seqs)
  if (n_fg == 0) {
    warning("No foreground sequences for method: ", method_label, " — skipping.")
    return(NULL)
  }
  
  fg_presence <- make_presence(fg_merged)
  
  all_motifs <- union(unique(fg_presence$motif_id),
                      unique(bg_presence$motif_id))
  
  results <- map_dfr(all_motifs, function(tf) {
    fisher_one_tf(
      motif           = tf,
      fg_seqs         = fg_seqs,
      bg_seqs         = bg_seqs,
      fg_present_seqs = fg_presence$sequence_name[fg_presence$motif_id == tf],
      bg_present_seqs = bg_presence$sequence_name[bg_presence$motif_id == tf],
      n_fg            = n_fg,
      n_bg            = n_bg
    )
  })
  
  if (nrow(results) == 0) return(NULL)
  
  results$method <- method_label
  results$padj   <- p.adjust(results$pvalue, method = "BH")
  results$freq_fg <- results$n_fg_hit / results$n_fg_total
  results$freq_bg <- results$n_bg_hit / results$n_bg_total
  results
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
  
  # Load FIMO hits
  message("Loading foreground FIMO: ", args$fimo_fg)
  fimo_fg <- load_fimo(args$fimo_fg, args$motif_col,
                       args$split_motif_id, tf_whitelist)
  message("  ", nrow(fimo_fg), " hits after filtering.")
  
  message("Loading background FIMO: ", args$fimo_bg)
  fimo_bg <- load_fimo(args$fimo_bg, args$motif_col,
                       args$split_motif_id, tf_whitelist)
  message("  ", nrow(fimo_bg), " hits after filtering.")
  
  # Load sequence tables
  message("Reading foreground sequences: ", args$seq_fg)
  seq_fg <- read_seq_table(args$seq_fg)
  message("  ", nrow(seq_fg), " sequences.")
  
  message("Reading background sequences: ", args$seq_bg)
  seq_bg <- read_seq_table(args$seq_bg)
  message("  ", nrow(seq_bg), " sequences.")
  
  # Merge FIMO hits with sequence lists
  fg_merged <- merge_fimo_with_seqs(fimo_fg, seq_fg)
  bg_merged <- merge_fimo_with_seqs(fimo_bg, seq_bg)
  
  # Background presence matrix (computed once)
  bg_presence <- make_presence(bg_merged)
  bg_seqs <- unique(seq_bg$sequence_name)
  n_bg <- length(bg_seqs)
  
  # Run Fisher per foreground method
  methods <- unique(fg_merged$method)
  message("Running Fisher's exact test for ", length(methods), " method(s)...")
  
  all_results <- map_dfr(methods, function(m) {
    message("  Processing: ", m)
    fg_sub <- fg_merged[fg_merged$method == m, ]
    run_fisher_one_method(fg_sub, bg_presence, bg_seqs, n_bg,
                          method_label = m)
  })
  
  # Select and order output columns
  out_cols <- c("method", "motif_id", "odds_ratio", "ci_low", "ci_high",
                "pvalue", "padj", "n_fg_hit", "n_fg_total", "freq_fg",
                "n_bg_hit", "n_bg_total", "freq_bg", "zero_cell")
  all_results <- all_results[, out_cols]
  all_results <- all_results[order(all_results$method, all_results$padj), ]
  
  write.table(all_results, file = args$output, sep = "\t", row.names = FALSE,
              quote = FALSE)
  message("\nResults written to: ", args$output)
  message(nrow(all_results), " tests across ", length(methods), " method(s) and ",
          length(unique(all_results$motif_id)), " TFs.")
}

main()
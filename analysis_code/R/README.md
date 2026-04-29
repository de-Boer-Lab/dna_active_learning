# TF Motif & Dinucleotide Enrichment Analysis

Three R scripts for quantifying sequence-feature enrichment between experimental and reference sequence sets.

## Dependencies

```r
install.packages(c("argparse", "stringr", "dplyr", "purrr", "effectsize"))
```

## Scripts

### 1. `tfbs_enrichment.R` — TF motif count enrichment (Cliff's delta)

Counts total FIMO motif hits per sequence, then computes Cliff's delta and AUROC between experimental and reference sets.

```bash
Rscript tfbs_enrichment.R \
  --fimo       <fimo.tsv or directory of fimo.tsv files> \
  --ref_seq    <reference_sequences.tsv> \
  --exp_seq    <experimental_sequences.tsv> \
  --output     <results.tsv> \
  [--motif_col       motif_alt_id]   # FIMO column for TF ID (default: motif_id)
  [--split_motif_id]                 # split on "." for HOCOMOCO IDs
  [--tf_whitelist     <tf_ids.txt>]  # one TF ID per line
```

**Output columns:** `method`, `cliffs_delta`, `cliffs_delta_CI_low`, `cliffs_delta_CI_high`, `auroc`, `n_exp`, `n_ref`

---

### 2. `dinucl_enrichment.R` — Dinucleotide content enrichment (Cliff's delta)

Computes per-dinucleotide Cliff's delta and AUROC from pre-computed frequency tables.

```bash
Rscript dinucl_enrichment.R \
  --exp_freq   <experimental_dinucl_freq.tsv> \
  --ref_freq   <reference_dinucl_freq.tsv> \
  --output     <results.tsv> \
  [--exp_labels  <sequences.tsv>]  # split experimental set by method
  [--n_dinucl    16]               # number of frequency columns (default: 16)
```

**Output columns:** `method`, `dinucleotide`, `cliffs_delta`, `cliffs_delta_CI_low`, `cliffs_delta_CI_high`, `auroc`, `n_exp`, `n_ref`

---

### 3. `tf_enrichment_fisher.R` — Per-TF enrichment (Fisher's exact test)

Tests each TF individually using a 2×2 presence/absence contingency table (Fisher's exact test, BH correction).

```bash
Rscript tf_enrichment_fisher.R \
  --fimo_fg    <fimo_foreground.tsv or directory> \
  --fimo_bg    <fimo_background.tsv or directory> \
  --seq_fg     <foreground_sequences.tsv> \
  --seq_bg     <background_sequences.tsv> \
  --output     <results.tsv> \
  [--motif_col       motif_alt_id]
  [--split_motif_id]
  [--tf_whitelist     <tf_ids.txt>]
```

**Output columns:** `method`, `motif_id`, `odds_ratio`, `ci_low`, `ci_high`, `pvalue`, `padj`, `n_fg_hit`, `n_fg_total`, `freq_fg`, `n_bg_hit`, `n_bg_total`, `freq_bg`, `zero_cell`

## Input formats

All sequence tables are **3-column TSV without header**:

```
sequence_name	sequence	method
```

`sequence_name` may be pipe-delimited (`seqid|method`); all scripts parse this automatically. When multiple methods are present in the experimental/foreground file, metrics are computed per method.

FIMO inputs are standard MEME Suite `fimo.tsv` output. A directory path can be passed instead of a single file — all `fimo.tsv` files found recursively will be concatenated (useful for per-chromosome runs).

Dinucleotide frequency tables are **TSV with header**, where the first *N* columns (default 16) are the dinucleotide frequencies, one row per sequence.
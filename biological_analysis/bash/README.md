# Analysis of sequences selected by active learning methods

The sequence analyses relies on two software packages: [polygraph](https://github.com/Genentech/polygraph) and the [MEME suite](https://meme-suite.org/). A conda environment is provided for both. `meme.yaml` and `polygraph.yaml` should be used to install the conda environments for those tools.

## Data management

Formating the data files to use as input for polygraph (tabular), fimo (fasta) and sea (fasta). The fasta to tab conversion relies on seqkit package that is contained in the meme conda environment.

### Formating input sequences

`bash biological_analysis/bash/formatFiles.sh pool.txt ALpool`

Conversion from fasta to a tabular file (`Seq.tsv`) containing the sequence ID (1st column), the nucleotide sequence (2nd column), and group the sequences belongs to (here "ALpool", 3rd column). Also produces a tabular file with no sequence ID for polygraph (`Seq.noID.tsv`) and a fasta file for fimo (`Seq.fasta`) with proper group names for downstream analysis.

Yeast data requires further trimming to remove the padding sequences added to the N80. Which is done by the `TrimmSequences.sh` script.

## Sequence analysis

### Di-nucleotide content

Di-nucleotide content is computed using the [polygraph](https://github.com/Genentech/polygraph) package.

`bash biological_analysis/bash/diNuclContent.sh Seq.noID.tsv diNuclContent.Seq.tsv`

### TF motifs scanning

`bash biological_analysis/bash/memeFimoScan.sh Seq.fasta TF_MOTIFS.meme OUT_FOLDER BACKGROUND.txt`

For both human and yeast, motif scanning is done only on the AL pool but not on the selected sequences directly because it is too computationally intensive otherwise.

A custom background file is generated with `fasta-get-markov ALpool.fasta` and passed to FIMO.
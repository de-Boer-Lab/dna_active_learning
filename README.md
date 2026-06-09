![nextFrag](assets/nextFrag_logo.png)

A PyTorch-based framework for benchmarking active learning (AL) strategies on DNA sequence-to-expression models. Implements three DREAM Challenge architectures (CNN, RNN, Attention) with several acquisition functions: MC Dropout, k-means, LCMD, and ensemble disagreement.

## Installation

**GPU (recommended):**

In addition to other dependencies, installs cuML for GPU-accelerated acquisition strategies and clustering (requires CUDA 12+ and Linux).

```bash
git clone https://github.com/de-Boer-Lab/nextFrag.git
cd nextFrag

# Option 1 (pip)
pip install -e ".[gpu]"

# Option 2 (conda)
conda create -n nextfrag -c rapidsai -c conda-forge -c nvidia \
    cuml=26.04 python=3.12 'cuda-version>=12.2,<=12.9'
conda activate nextfrag
pip install -e .
```

**CPU-only:**
```bash
git clone https://github.com/de-Boer-Lab/nextFrag.git
cd nextFrag
pip install -e .
```

## Dataset Setup

Before running experiments, use `setup_dataset.sh` to scaffold the expected directory layout, place your data files, initialize a `results.tsv` for tracking runs, and record the project root so all Python commands can find it.

Full datasets used in the paper are available on [Zenodo](https://zenodo.org/records/20094954).
Demo subsets for quick testing are included under `data/yeast/` and `data/human/`.

```bash
./setup_dataset.sh \
    --root /path/to/project_root \
    --dataset yeast \
    --train train.txt --pool pool.txt --val val.txt --test test.txt \
    --n-selected 20000
```

For the human dataset, pass additional test files:
```bash
./setup_dataset.sh \
    --root /path/to/project_root \
    --dataset human \
    --train train.txt --pool pool.txt --val val.txt --test test.txt \
    --test-ood test_ood.txt --test-vep test_vep.txt \
    --n-selected 20000
```

Files are copied by default. Use `--mode symlink` or `--mode move` to change this. Re-running is safe; existing files are left alone unless `--force` is passed.

**`--n-selected`** records a default selection size. Runs using that size will use the plain strategy name in directory paths (e.g. `mcd/`); runs with any other size append the count (e.g. `mcd_10k/`).

**Data format:** All data files are tab-separated with two columns: `sequence` and `expression_value`. Sequences should be raw nucleotide strings (A/C/G/T/N); preprocessing is handled automatically per dataset.

## Repository Overview

```
src/nextfrag/
├── config.py             # Dataset/architecture defaults; reads project root from ~/.config/nextfrag/
├── path_resolver.py      # Resolves all on-disk paths for a (dataset, round, arch, seed) run
├── utils.py              # Shared utilities
├── data/
│   └── dataloader.py     # Data loading, one-hot encoding, reverse complement handling
├── models/
│   ├── dream_models.py   # CNN, RNN, Attention architectures
│   ├── blocks.py         # Reusable network building blocks
│   ├── model_loader.py   # Model instantiation and checkpoint loading
│   ├── trainer.py        # Training loop
│   ├── train_model.py    # CLI for training
│   └── evaluation.py     # Test-set evaluation and results.tsv logging
├── acquisition/
│   ├── mc_dropout.py     # Uncertainty-based selection via Monte Carlo Dropout
│   ├── diversity.py      # Diversity-based selection (k-means, LCMD) with incremental PCA
│   ├── ensemble.py       # Disagreement-based selection across multiple models
│   └── biological.py     # Expression-based selection; in-silico saturation mutagenesis (ISM)
├── active_learning/
│   ├── loop.py           # Main AL loop: select → update data → train
│   └── utils.py          # Train/pool dataset update between rounds
└── analysis/
    ├── prediction_error.py  # Inference on a file; writes prediction error alongside ground truth
    └── umap.py              # UMAP embedding of pool sequences in model feature space

biological_analysis/         # Post-hoc sequence analyses, independent of the AL loop
├── python/
│   ├── gc_content.py        # GC content calculation
│   ├── getDiNuclContent.py  # Dinucleotide frequency analysis
│   ├── TF_motif_scanning.py # JASPAR TF motif scanning
│   └── filterWB.py          # Filter TFs by GTEx Whole Blood expression
├── bash/                    # Shell scripts for MEME/FIMO scanning and sequence trimming
└── R/                       # Enrichment analyses (TFBS content, dinucleotide content)
```

## Directory Structure

The AL experiments write to this on-disk layout:

```
{PROJECT_ROOT}/
└── {dataset}/
    ├── round_0/
    │   ├── data/                          # Initial training and pool data
    │   │   ├── train.txt
    │   │   └── pool.txt
    │   └── {arch}_{seed}/model/           # Round-0 model checkpoints
    │       └── model_best.pth
    ├── round_{i}/{acquisition}/{arch}_{seed}/
    │   ├── data/
    │   │   ├── selected.txt               # Sequences selected this round
    │   │   ├── train.txt                  # Cumulative training set
    │   │   └── pool.txt                   # Remaining unlabeled pool
    │   └── model/
    │       └── model_best.pth
    ├── val.txt
    ├── test.txt
    └── results.tsv                        # One row per trained model; appended automatically
```

## Usage

### Active Learning Loop

`nextfrag.active_learning.loop` runs a full multi-round AL experiment for **non-ensemble** strategies (`mcd`, `kmeans`, `lcmd`). The basic workflow is: run `setup_dataset.sh` once to scaffold directories and place data, train a round-0 model (manually or with `--train-initial-model`), then run the loop for subsequent rounds.

```bash
# Run rounds 1–3 starting from an existing round-0 model
python -m nextfrag.active_learning.loop \
    --dataset yeast --acquisition mcd --arch rnn --seed 42 \
    --num-rounds 3 --n-selected 20000

# Or train the round-0 model and run all rounds in one command
python -m nextfrag.active_learning.loop \
    --dataset yeast --acquisition mcd --arch rnn --seed 42 \
    --num-rounds 3 --n-selected 20000 --train-initial-model
```

`--acquisition` accepts `mcd` (MC Dropout), `kmeans`, or `lcmd`.

Ensemble strategies (`all_arch`, `same_arch`) require multiple models trained in parallel each round — see below.

### Ensemble Active Learning Loop

`ensemble_al_loop.sh` is an example SLURM orchestrator and analogue of `nextfrag.active_learning.loop` for ensemble AL. It invokes two helper scripts (`ensemble_select_and_update.sh`, `ensemble_train.sh`) to handle the per-round acquisition and per-model training respectively.

```bash
# Multi-architecture ensemble
bash ensemble_al_loop.sh yeast all_arch "rnn:1 cnn:2 attn:3"

# Same architecture, multiple replicates
bash ensemble_al_loop.sh yeast same_arch "rnn:1 rnn:2 rnn:3"
```

Arguments: dataset, acquisition strategy, and a space-separated list of `arch:seed` pairs. Data files are written under the first model's directory and shared across ensemble members within a round.

These scripts are written for SLURM — adapt resource flags and environment setup to your cluster.

### Train a Model

**Within an AL experiment** (paths resolved automatically):
```bash
python -m nextfrag.models.train_model al \
    --dataset yeast --arch cnn \
    --acquisition mcd --round 1 --seed 42
```

**With explicit paths** (standalone use):
```bash
python -m nextfrag.models.train_model custom \
    --dataset human --arch attn \
    --train data/train.txt --val data/val.txt --model-dir outputs/
```

Both modes evaluate the trained model and record results. To use a custom `nn.Module`, pass `--model-class mypackage.models.MyModel` (dotted import path).

### Sequence Selection

Each strategy can be run independently to select with previously-trained models.

**MC Dropout:**
```bash
python -m nextfrag.acquisition.mc_dropout \
    --dataset yeast --arch rnn --round 2 --seed 42 --num-passes 10
```

**Diversity (k-means or LCMD):**
```bash
python -m nextfrag.acquisition.diversity \
    --dataset yeast --arch rnn --acquisition kmeans --round 2 --seed 42
```

**Ensemble:**
```bash
# Multi-architecture ensemble
python -m nextfrag.acquisition.ensemble \
    --dataset human --acquisition all_arch --round 2 \
    --models rnn:1 cnn:2 attn:3

# Same-architecture ensemble (multiple replicates)
python -m nextfrag.acquisition.ensemble \
    --dataset human --acquisition same_arch --round 2 \
    --arch rnn --seeds 1 2 3 4 5
```

**Biological (expression-based):**
```bash
# Select highest-predicted sequences from the pool
python -m nextfrag.acquisition.biological \
    --dataset yeast --arch rnn --round 1 --seed 42 \
    --acquisition max_expr

# Select lowest-predicted sequences
python -m nextfrag.acquisition.biological \
    --dataset yeast --arch rnn --round 1 --seed 42 \
    --acquisition min_expr

# In-silico saturation mutagenesis on an explicit file
python -m nextfrag.acquisition.biological \
    --dataset yeast --arch rnn --seed 42 \
    --ism --file-path pool.txt --out-path ism_scores.tsv
```

### Evaluate a Model

Evaluate a trained model on held-out test data and append results to `results.tsv`:

```bash
python -m nextfrag.models.evaluation \
    --dataset yeast --arch rnn \
    --acquisition mcd --round 2 --seed 42
```

### Additional Analysis Tools

**Prediction error** — run inference and write prediction error alongside ground truth:
```bash
python -m nextfrag.analysis.prediction_error \
    --data-path data.txt --out-path preds.tsv \
    --dataset yeast --arch rnn --model-path model_best.pth
```

**UMAP** (GPU required; produces a TSV with `seq`, `expr`, `umap0`, `umap1` columns):
```bash
python -m nextfrag.analysis.umap \
    --dataset yeast --arch rnn \
    --model-path model_best.pth --out-path umap.tsv
```

## Biological Analyses

Scripts to perform biological analyses of AL selected sequences compared to the overall AL pool. Links to documentation:

[Input sequences formatting](https://github.com/de-Boer-Lab/nextFrag/tree/main/biological_analysis/bash)

[Computing of dinucleotide content and TFBS scanning](https://github.com/de-Boer-Lab/nextFrag/tree/main/biological_analysis/bash)

[Enrichment analyses: dinucleotide content, TFBS content and TFs](https://github.com/de-Boer-Lab/nextFrag/tree/main/biological_analysis/R)

## Adding New Architectures

Register a new architecture for running AL experiments in three steps:

1. Define `_init_myarch_model(dataset)` in [model_loader.py](src/nextfrag/models/model_loader.py)
2. Add `case "myarch":` to `init_model()`'s match statement
3. Add training defaults to `ARCH_CONFIG` in [config.py](src/nextfrag/config.py)

Alternatively, pass `--model-class mypackage.models.MyModel` to the training CLI to use any `nn.Module` directly without modifying the registry.

# Paper

Check out our [preprint on bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.21.727038v1) titled, "*Evaluation of Active Learning Selection Strategies and Characterization of Informative Sequences for Sequence-to-Expression Models*", for more details.


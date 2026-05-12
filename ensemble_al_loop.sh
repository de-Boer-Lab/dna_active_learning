#!/bin/bash

DATASET=$1
ACQ=$2
MODELS=$3 # e.g. "rnn:1 cnn:2 attn:3"

ROUNDS=3
SELECT_AND_UPDATE_SCRIPT=ensemble_select_and_update.sh
TRAIN_SCRIPT=ensemble_train.sh

read -ra MODEL_ARR <<< "$MODELS"
N_MODELS=${#MODEL_ARR[@]}

if [[ $N_MODELS -eq 0 ]]; then
    echo "ERROR: MODELS argument is empty" >&2
    exit 1
fi

FIRST_ARCH_SEED=${MODEL_ARR[0]}              # e.g. "rnn:1"
FIRST_MODEL_PATH=${FIRST_ARCH_SEED/:/_}      # e.g. "rnn_1"

echo "Parsed $N_MODELS models: ${MODEL_ARR[*]}"
echo "FIRST_MODEL_PATH=$FIRST_MODEL_PATH"

select_deps=""

for r in $(seq 1 $ROUNDS); do
    echo "=== Submitting round $r ==="

    export ROUND=$r
    export NUM_ROUNDS=$ROUNDS
    export DATASET ACQ MODELS FIRST_MODEL_PATH

    # Select and update
    jid_select=$(sbatch --parsable \
        $select_deps \
        --export=ALL \
        $SELECT_AND_UPDATE_SCRIPT)
    echo "Submitted select: jid=$jid_select"

    # Train
    jid_train=$(sbatch --parsable \
        --array=1-$N_MODELS \
        --dependency=afterok:$jid_select \
        --export=ALL \
        $TRAIN_SCRIPT)
    echo "Submitted train: jid=$jid_train"

    select_deps="--dependency=afterok:$jid_train"
done
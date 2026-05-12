#!/bin/bash

read -ra MODEL_ARR <<< "$MODELS"

IDX=$((SLURM_ARRAY_TASK_ID - 1))
PAIR=${MODEL_ARR[$IDX]}

if [[ -z "$PAIR" ]]; then
    echo "ERROR: no model at index $IDX (MODELS=$MODELS)" >&2
    exit 1
fi

ARCH=${PAIR%:*}
SEED=${PAIR#*:}

echo "Array task $SLURM_ARRAY_TASK_ID: ARCH=$ARCH SEED=$SEED"

source ~/.bashrc
conda activate nextfrag
python -m nextfrag.models.train_model al \
    --dataset $DATASET --arch $ARCH --acquisition $ACQ \
    --round $ROUND --seed $SEED

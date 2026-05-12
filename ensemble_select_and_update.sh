#!/bin/bash

# select
source ~/.bashrc
conda activate nextfrag
python -m nextfrag.acquisition.ensemble \
    --dataset $DATASET --acquisition $ACQ --round $ROUND --models $MODELS

PREV_ROUND=$(($ROUND - 1))

# update
cd "$(cat ~/.config/nextfrag/root)/${DATASET}"
if [[ $ROUND -gt 1 ]]; then
    cat round_$PREV_ROUND/$ACQ/$FIRST_MODEL_PATH/data/train.txt round_$ROUND/$ACQ/$FIRST_MODEL_PATH/data/selected.txt > round_$ROUND/$ACQ/$FIRST_MODEL_PATH/data/train.txt

    if [[ $ROUND -lt $NUM_ROUNDS ]]; then
        grep -F -x -v -f round_$ROUND/$ACQ/$FIRST_MODEL_PATH/data/selected.txt round_$PREV_ROUND/$ACQ/$FIRST_MODEL_PATH/data/pool.txt  > round_$ROUND/$ACQ/$FIRST_MODEL_PATH/data/pool.txt
    fi

    rm round_$PREV_ROUND/$ACQ/$FIRST_MODEL_PATH/data/train.txt round_$PREV_ROUND/$ACQ/$FIRST_MODEL_PATH/data/pool.txt
else # first round
    cat round_0/data/train.txt round_1/$ACQ/$FIRST_MODEL_PATH/data/selected.txt > round_1/$ACQ/$FIRST_MODEL_PATH/data/train.txt

    if [[ $ROUND -lt $NUM_ROUNDS ]]; then
        grep -F -x -v -f round_1/$ACQ/$FIRST_MODEL_PATH/data/selected.txt round_0/data/pool.txt  > round_1/$ACQ/$FIRST_MODEL_PATH/data/pool.txt
    fi
fi

echo "Done!"

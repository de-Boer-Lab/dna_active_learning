#!/bin/bash

# Load conda environment
source ~/.bashrc
conda activate meme

INFASTA=$1
MOTIFS=$2
OUTDIR=$3
BACKGROUND=$4


FDR=0.01 #Set FDR low, might need to go lower when number of sequences increases

DIRNAME=$(echo ${OUTDIR}\_FDR0.01)

# Check if background file is provided
if [ -z "$BACKGROUND" ] || [ ! -f "$BACKGROUND" ]; then
    echo "No background file provided or file does not exist. Generating background file..."
    BCKGND=$(basename ${INFASTA} .fasta).bckgnd
    fasta-get-markov ${INFASTA} > ${BCKGND}
else
    echo "Using provided background file: ${BACKGROUND}"
    BCKGND=${BACKGROUND}
fi

echo "Scanning for motifs"
fimo --bfile ${BCKGND} \
--max-strand \
--thresh ${FDR} \
--qv-thresh \
--verbosity 1 \
--o ${DIRNAME} ${MOTIFS} ${INFASTA}

cd ${DIRNAME}

rm fimo.html fimo.gff fimo.xml cisml.xml #Removing the useless files taking up a lot of space

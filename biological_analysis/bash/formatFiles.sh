#!/bin/bash

# Load conda environment
source ~/.bashrc
conda activate meme

#This script convert the tab sequence format orginally used with polygraph for motif scanning with the meme suite because there is a loss of sequence information 
# with polygraph

INFASTA=$1
GROUP=$2

FILENAME=$(basename ${INFASTA} .txt)

seqkit fx2tab ${INFASTA} | awk -v OFS="\t" -v group=${GROUP} '{print $1,$2,group}'> ${FILENAME}.tsv
cut -f 2,3 ${FILENAME}.tsv > ${FILENAME}.noID.tsv
#Add the metadata (e.g. the model architecture, the AL selection method and the seed number) of each sequence to the sequence name  of the fasta file
awk -v OFS="\t" '{print $1"|"$3, $2}' ${FILENAME}.tsv | seqkit tab2fx > ${FILENAME}.fasta

conda deactivate
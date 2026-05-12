#!/bin/bash

source ~/.bashrc
conda activate meme

#This script trimms the padding added to each sequence of the yeast dataset and produces the files that will be used for downstream analysis

INFILE=$1

FILENAME=$(basename ${INFILE} .tsv)
# Trim sequences: skip the first 57 nt of upstream plasmid context and keep the 80 nt insert
awk -v OFS="\t" '{print $1, substr($2, 58, 80), $3}' ${INFILE} > ${FILENAME}.trim.tsv

#Remove the ID column for polygraph input
cut -f 2,3 ${FILENAME}.trim.tsv > ${FILENAME}.trim.noID.tsv

#Fasta sequence
awk -v OFS="\t" '{print $1"|"$3, $2}' ${FILENAME}.trim.tsv | seqkit tab2fx > ${FILENAME}.trim.fasta

conda deactivate 

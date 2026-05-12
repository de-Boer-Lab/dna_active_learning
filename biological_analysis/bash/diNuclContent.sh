#!/bin/bash

source ~/.bashrc
conda activate polygraph

PYSCRIPT="$(dirname "$0")/../python/getDiNuclContent.py"

INTAB=$1
OUTFILE=$2

python ${PYSCRIPT} ${INTAB} ${OUTFILE}
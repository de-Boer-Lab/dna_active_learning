#!/bin/bash

#For Numba and matplotlib caching 

export MPLCONFIGDIR=/cache/matplotlib
export NUMBA_CACHE_DIR=/cache/numba

source ~/.bashrc
conda activate polygraph

PYSCRIPT="/python/getDiNuclContent.py"

INTAB=$1
OUTFILE=$2

python ${PYSCRIPT} ${INTAB} ${OUTFILE}
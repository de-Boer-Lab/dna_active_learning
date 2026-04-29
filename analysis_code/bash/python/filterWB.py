#! python3

import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
import matplotlib.pyplot as plt
import sys

##This scripts filters for whole blood TF following the counts per sequence from polygraph
sites = pd.read_csv('/scratch/st-cdeboer-1/emmanuel/al_selected_seqs/human/jaspar.pooled.deduplicated.tsv', sep='\t')
gtex = pd.read_csv('/scratch/st-cdeboer-1/emmanuel/al_selected_seqs/human/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz', sep='\t', header =2)

# Get blood-expressed TFs (this line is probably correct)
blood_tfs = gtex.loc[gtex['Whole Blood'] > 1, 'Description']

# Get all TF names from column headers (excluding 'SeqID')
all_tfs = pd.Index(sites.columns[1:]).str.upper()  # Skip SeqID column

# Find intersection
selected_tfs = set(all_tfs).intersection(blood_tfs.str.upper())

# Filter sites DataFrame to only keep columns for selected TFs
# Keep SeqID column plus selected TF columns
cols_to_keep = ['SeqID'] + [col for col in sites.columns[1:] if col.upper() in selected_tfs]
sites = sites[cols_to_keep]

sites.to_csv('jaspar.all.filteredWholeBlood.tsv', header = True, sep = '\t')
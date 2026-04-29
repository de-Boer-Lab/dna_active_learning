#! python3

import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import polygraph.input, polygraph.embedding, polygraph.motifs, polygraph.models, polygraph.sequence, polygraph.utils, polygraph.visualize, polygraph.stats
pd.set_option('display.precision', 2)

selected_sequences = sys.argv[1]

seqs = polygraph.input.read_seqs(selected_sequences)

sites = polygraph.motifs.scan(seqs, '/scratch/st-cdeboer-1/emmanuel/al_selected_seqs/human/tf_motif/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt', pthresh=0.001)

#Filter on Whole Blood TF because only K562 sequences
gtex = pd.read_csv('/scratch/st-cdeboer-1/emmanuel/al_selected_seqs/human/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz', sep='\t', header =2)
blood_tfs = gtex.loc[gtex['Whole Blood'] > 1, 'Description']
all_tfs = sites.MotifID.str.upper()
selected_tfs = set(all_tfs).intersection(blood_tfs)
sites = sites[sites.MotifID.isin(selected_tfs)]

#counts = polygraph.motifs.motif_frequencies(sites, seqs=seqs, normalize=False)

sites.to_csv('jaspar.all.sites.filteredWholeBlood.tsv', header = True, index = False, sep = '\t')
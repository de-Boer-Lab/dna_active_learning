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
out_name = sys.argv[2]

seqs = polygraph.input.read_seqs(selected_sequences)

#kmers = polygraph.sequence.kmer_frequencies(seqs=seqs, k=1, normalize=True)
kmers_normalized = polygraph.sequence.kmer_frequencies(seqs=seqs, k=2, normalize=True)

#kmers.to_csv('MonoNuclContentNormalized.9-30918045-T-C-A-wC.tsv', header = True, index = False, sep = '\t')
kmers_normalized.to_csv(out_name, header = True, index = False, sep = '\t')
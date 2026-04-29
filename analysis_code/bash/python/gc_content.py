#! python3

import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import polygraph.input, polygraph.embedding, polygraph.motifs, polygraph.models, polygraph.sequence, polygraph.utils, polygraph.visualize, polygraph.stats
pd.set_option('display.precision', 2)
#%matplotlib inline

selected_sequences = sys.argv[1]
out_name = sys.argv[2]

seqs = polygraph.input.read_seqs(selected_sequences)

seqs['GC Content'] = polygraph.sequence.gc(seqs)
out_gc = seqs.iloc[:, 1:3]
#test_results = polygraph.stats.kruskal_dunn(data=seqs, val_col="GC Content")

out_gc.to_csv(out_name, header = True, index = False, sep = '\t')
#test_results.to_csv('gc_content.stats.tsv', header = True, index = False, sep = '\t')
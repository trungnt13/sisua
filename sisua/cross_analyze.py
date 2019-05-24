from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle

import seaborn as sns
from matplotlib import pyplot as plt
from odin import visual as vs
from odin.ml import fast_tsne, fast_pca

import numpy as np

from sisua.analysis import Posterior
from sisua.data.utils import standardize_protein_name
from sisua.data import CROSS8k_lymphoid, CROSSecc_lymphoid

# ===========================================================================
# Load datasets
# ===========================================================================
ds = CROSS8k_lymphoid()
ds8k = {
    'X': ds['X'][:],
    'X_col': ds['X_col'],
    'X_row': ds['X_row'],
    'y': ds['y'],
    'y_col': [standardize_protein_name(i) for i in ds['y_col']],
}

ds = CROSSecc_lymphoid()
dsecc = {
    'X': ds['X'][:],
    'X_col': ds['X_col'],
    'X_row': ds['X_row'],
    'y': ds['y'],
    'y_col': [standardize_protein_name(i) for i in ds['y_col']],
}

all_proteins = set(ds8k['y_col']) & set(dsecc['y_col'])
print("Share proteins:", all_proteins)

ids = [i for i, j in enumerate(ds8k['y_col']) if j in all_proteins]
ds8k['y'] = ds8k['y'][:, ids]
ds8k['y_col'] = np.array(ds8k['y_col'])[ids]

ids = [i for i, j in enumerate(dsecc['y_col']) if j in all_proteins]
dsecc['y'] = dsecc['y'][:, ids]
dsecc['y_col'] = np.array(dsecc['y_col'])[ids]

# ===========================================================================
# Load inferences
# ===========================================================================
model_name = 'movae_genecell_Xlog0zinb_Yprob0bernoulli_Traw_Znormal_080spvs1e+02_net02128032_drop30000000_1kl400_alytcT_bnormT_binomial25'
with open(os.path.join('/home/trung/bio_log/crossecc_ly', model_name, 'model.pkl'), 'rb') as f:
  infer_pbmcecc = pickle.load(f)
with open(os.path.join('/home/trung/bio_log/cross8k_ly', model_name, 'model.pkl'), 'rb') as f:
  infer_pbmc8k = pickle.load(f)

# ===========================================================================
# Analysis
# ===========================================================================
outpath = '/tmp/cross'
if os.path.exists(outpath):
  shutil.rmtree(outpath)
os.mkdir(outpath)

Posterior(infer_pbmcecc, ds=ds8k
).new_figure(
).plot_latents_scatter(size=4,
).plot_latents_heatmap(
).plot_correlation_series(
).plot_protein_series(
).save_plots(os.path.join(outpath, 'modelECC_ds8k.pdf'))

Posterior(infer_pbmcecc, ds=dsecc
).new_figure(
).plot_latents_scatter(size=4
).plot_latents_heatmap(
).plot_correlation_series(
).plot_protein_series(
).save_plots(os.path.join(outpath, 'modelECC_dsECC.pdf'))

Posterior(infer_pbmc8k, ds=ds8k
).new_figure(
).plot_latents_scatter(size=4
).plot_latents_heatmap(
).plot_correlation_series(
).plot_protein_series(
).save_plots(os.path.join(outpath, 'model8k_ds8k.pdf'))

Posterior(infer_pbmc8k, ds=dsecc
).new_figure(
).plot_latents_scatter(size=4
).plot_latents_heatmap(
).plot_correlation_series(
).plot_protein_series(
).save_plots(os.path.join(outpath, 'model8k_dsECC.pdf'))

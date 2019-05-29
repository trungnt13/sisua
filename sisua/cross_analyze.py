from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle
from six import string_types

import seaborn as sns
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.ml import fast_tsne, fast_pca
from odin.utils import as_tuple, ctext

import numpy as np

from sisua.data import get_dataset, get_dataset_summary
from sisua.analysis import Posterior
from sisua.data.utils import standardize_protein_name


def cross_analyze(datasets, models, verbose=False):
  datasets = as_tuple(datasets, t=string_types)
  models = as_tuple(models, t=string_types)
  assert len(datasets) > 1, \
  "Require more than one datasets for cross analysis"
  # ====== load datasets ====== #
  all_datasets = {name: get_dataset(name)[0]
                  for name in datasets}
  all_datasets = [
  (name, dict(X=ds['X'][:],
              X_col=ds['X_col'],
              X_row=ds['X_row'],
              y=ds['y'],
              y_col=np.array([standardize_protein_name(i)
                              for i in ds['y_col']]),
  ))
  for name, ds in all_datasets.items()]
  # ====== get the list of all overlapping protein ====== #
  all_proteins = set(all_datasets[0][1]['y_col'])
  for name, ds in all_datasets:
    all_proteins &= set(ds['y_col'])
  if verbose:
    print("Datasets       :", ctext(', '.join(datasets), 'yellow'))
    print("Models         :", ctext(', '.join(models), 'yellow'))
    print("Shared proteins:", ctext(', '.join(all_proteins), 'yellow'))
  exit()

  ids = [i for i, j in enumerate(ds8k['y_col']) if j in all_proteins]
  ds8k['y'] = ds8k['y'][:, ids]
  ds8k['y_col'] = np.array(ds8k['y_col'])[ids]

  ids = [i for i, j in enumerate(dsecc['y_col']) if j in all_proteins]
  dsecc['y'] = dsecc['y'][:, ids]
  dsecc['y_col'] = np.array(dsecc['y_col'])[ids]

cross_analyze(datasets=['cross8k_ly', 'crossecc_ly'],
              models='vae',
              verbose=True)

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

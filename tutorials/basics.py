from __future__ import print_function, division, absolute_import

import os
import pickle

# for calculation
import numpy as np
import tensorflow as tf

# for visualization
import seaborn as sns
from odin import visual as vs
from matplotlib import pyplot as plt

# SISUA API
from sisua.data import get_dataset
from sisua.inference import Inference
from sisua.analysis import Posterior

# ===========================================================================
# Configurations
# ===========================================================================
SAVE_FIGURE_PATH = '/tmp/tmp.pdf'
SAVE_DATA_FIGURE_PATH = '/tmp/data.pdf'
SAVE_MODEL_PATH = '/tmp/model.pkl'

corruption_rate = 0.25
corruption_dist = 'binomial'
# ===========================================================================
# Loading data
# ===========================================================================
dataset, gene_ds, prot_ds = get_dataset('pbmc8k_ly')
print(dataset)

gene_ds.plot_percentile_histogram(n_hist=8, title="Gene")
prot_ds.plot_percentile_histogram(n_hist=8, title="Protein")
vs.plot_save(SAVE_DATA_FIGURE_PATH)

# ====== get train and test data ====== #
# the SingleCellDataset will ensure data splitting is
# consistent every time running the experiment
X_train = gene_ds.get_data(data_type='train')
y_train = prot_ds.get_data(data_type='train')

n_genes = X_train.shape[1]
n_prot = y_train.shape[1]

# ===========================================================================
# Training
# ===========================================================================
# simple sklearn API style
infer = Inference(gene_dim=n_genes, prot_dim=n_prot,
                  model='kmovae', dispersion='gene-cell',
                  xnorm='log', tnorm='raw', ynorm='prob',
                  xclip=0, yclip=0,
                  xdist='zinb', ydist='bernoulli', zdist='normal',
                  xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
                  hdim=128, zdim=32, nlayer=2,
                  batchnorm=True, analytic=True,
                  kl_weight=1., warmup=400, y_weight=1.)
infer.fit(X=X_train, y=y_train,
          supervised_percent=0.8, validation_percent=0.1,
          n_mcmc_samples=1,
          corruption_rate=0.25, corruption_dist='binomial',
          batch_size=128, n_epoch=120, learning_rate=1e-4,
          detail_logging=True)
# ====== the model can be saved simply using pickle ====== #
with open(SAVE_MODEL_PATH, 'wb') as f:
  pickle.dump(infer, f)

# ===========================================================================
# Evaluation using Posterior
# ===========================================================================
# path to saved model can be provided
# instead of Inference as well
pos = Posterior(infer, ds=dataset)
# chaining methods for convenience
# first evaluate the latents
pos.new_figure(
).plot_latents_scatter(
).plot_latents_heatmap(
).plot_streamline_F1(mode='ovr'
).plot_streamline_F1(mode='ovo')
# second evaluate the imputation
pos.plot_correlation_series(
).new_figure(nrow=8, ncol=12
).plot_cellsize_series(test=False, ax=211
).plot_cellsize_series(test=True, ax=212)
# third the protein prediction
pos.plot_protein_series(
).plot_protein_scatter()
# Finally debugging the training process
pos.new_figure(
).plot_learning_curves()
# all generated plot can be saved fast
pos.save_plots(SAVE_FIGURE_PATH)

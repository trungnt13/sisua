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
from sisua.analysis import Posterior, ResultsSheet
from sisua.inference import (InferenceSCVAE, InferenceDCA,
                             InferenceSCVI, InferenceSISUA)

# ===========================================================================
# Configurations
# ===========================================================================
SAVE_FOLDER = '/tmp/sisua_basics'
if not os.path.exists(SAVE_FOLDER):
  os.mkdir(SAVE_FOLDER)

SAVE_DATA_FIGURE_PATH = os.path.join(SAVE_FOLDER, 'data.pdf')

SAVE_FIGURE_PATH = os.path.join(SAVE_FOLDER, 'sisua_figure.pdf')
SAVE_SCORES_PATH = os.path.join(SAVE_FOLDER, 'sisua_scores.html')

SAVE_COMPARE_FIGURE_PATH = os.path.join(SAVE_FOLDER, 'compare_figure.pdf')
SAVE_COMPARE_SCORES_PATH = os.path.join(SAVE_FOLDER, 'compare_scores.html')

SAVE_MODEL_PATH = {
    'sisua': os.path.join(SAVE_FOLDER, 'sisua.pkl'),
    'scvi': os.path.join(SAVE_FOLDER, 'scvi.pkl'),
    'scvae': os.path.join(SAVE_FOLDER, 'scvae.pkl'),
    'dca': os.path.join(SAVE_FOLDER, 'dca.pkl'),
}

n_epoch = 200
# ===========================================================================
# Loading data
# ===========================================================================
dataset, gene_ds, prot_ds = get_dataset('pbmc8k_my')
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
# Training different models
# ===========================================================================
all_models = {}
for name in ('sisua', 'scvi', 'scvae', 'dca'):
  print("\nStart training %s ..." % name)
  # simple sklearn API style, single process training
  if name == 'sisua':
    model = InferenceSISUA(gene_dim=n_genes, prot_dim=n_prot)
  elif name == 'scvi':
    model = InferenceSCVI(gene_dim=n_genes)
  elif name == 'scvae':
    model = InferenceSCVAE(gene_dim=n_genes)
  elif name == 'dca':
    model = InferenceDCA(gene_dim=n_genes)
  else:
    raise NotImplementedError
  # All models are the same, fast and easy
  model.fit(X_train,
            y=y_train if model.is_semi_supervised else None,
            n_epoch=n_epoch, detail_logging=True)
  # the model can be saved simply using pickle
  with open(SAVE_MODEL_PATH[name], 'wb') as f:
    pickle.dump(model, f)
  print("Saved trained model to: ", SAVE_MODEL_PATH[name])
  # keep the model for later use
  all_models[name] = model

# ===========================================================================
# Evaluation using Posterior
# ===========================================================================
# path to saved model can be provided or instance of Inference as well
all_posteriors = {
    name: Posterior(model, ds=dataset)
    for name, model in all_models.items()
}
# ====== just evaluate SISUA ====== #
pos = all_posteriors['sisua']
# chaining methods for convenience
# first evaluate the latents
pos.new_figure(
).plot_latents_scatter(
).plot_latents_heatmap(
).plot_streamline_F1(mode='ovr'
).plot_streamline_F1(mode='ovo'
)
# second evaluate the imputation
pos.plot_correlation_series(
).new_figure(nrow=8, ncol=12
).plot_cellsize_series(test=False, ax=211
).plot_cellsize_series(test=True, ax=212
)
# third the protein prediction
pos.plot_protein_series(
).plot_protein_scatter(
)
# Finally debugging the training process
pos.new_figure(
).plot_learning_curves(
)
# all generated plot can be saved fast
pos.save_plots(SAVE_FIGURE_PATH
).save_scores(SAVE_SCORES_PATH)
# ===========================================================================
# Comparison multiple systems
# ===========================================================================
# Create a combined analysis
analyzer = ResultsSheet(*list(all_posteriors.values()))

# same strategy, method chaining
analyzer.plot_correlation_series(
).plot_latents_scatter(pca=False
).plot_imputation_scatter(color_by_library=False
).plot_imputation_scatter(color_by_library=True
).plot_scores(score_type='pearson'
).plot_scores(score_type='cluster'
).plot_scores(score_type='classifier'
).save_plots(SAVE_COMPARE_FIGURE_PATH
).save_scores(SAVE_COMPARE_SCORES_PATH)

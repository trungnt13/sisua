# TODO fig bug here
from __future__ import print_function, division, absolute_import

import os
import time
import pickle
from multiprocessing import Pool

# for visualization
from odin.utils import ctext
from odin import visual as vs

# SISUA API
from sisua.data import get_dataset
from sisua.analysis import ResultsSheet, Posterior

# ===========================================================================
# Configurations
# ===========================================================================
SAVE_FOLDER = '/tmp/sisua_basics'
if not os.path.exists(SAVE_FOLDER):
  os.mkdir(SAVE_FOLDER)

SAVE_DATA_FIGURE_PATH = os.path.join(SAVE_FOLDER, 'data.pdf')

SAVE_FIGURE_PATH = SAVE_FOLDER

SAVE_COMPARE_FIGURE_PATH = os.path.join(SAVE_FOLDER, 'compare_figure.pdf')
SAVE_COMPARE_SCORES_PATH = os.path.join(SAVE_FOLDER, 'compare_scores.html')

all_models = ('sisua', 'scvi', 'scvinl', 'scvae', 'dca')
SAVE_MODEL_PATH = {i: os.path.join('/tmp', i)
                   for i in all_models}
n_epoch = 1
# ===========================================================================
# Loading data
# ===========================================================================
dataset, gene_ds, prot_ds = get_dataset('pbmc8k_ly')
print(dataset)
eval_ds = dict(
    X_train=gene_ds.X_train, X_test=gene_ds.X_test, X_col = gene_ds.col_name,
    y_train=prot_ds.X_train, y_test=prot_ds.X_test, y_col = prot_ds.col_name,
)

gene_ds.plot_percentile_histogram(n_hist=8, title="Gene")
prot_ds.plot_percentile_histogram(n_hist=8, title="Protein")
vs.plot_save(SAVE_DATA_FIGURE_PATH)

# ====== get train and test data ====== #
# the SingleCellOMIC will ensure data splitting is
# consistent every time running the experiment
X_train = gene_ds.get_data(data_type='train')
y_train = prot_ds.get_data(data_type='train')

n_genes = X_train.shape[1]
n_prot = y_train.shape[1]

# ===========================================================================
# Training different models
# ===========================================================================
def train_and_return(name):
  from sisua.inference import (InferenceSCVAE, InferenceDCA,
                               InferenceSCVI, InferenceSISUA)
  print("Start training %s ..." % ctext(name, 'lightyellow'))

  start_time = time.time()
  # simple sklearn API style, single process training
  if name == 'sisua':
    model = InferenceSISUA(gene_dim=n_genes, prot_dim=n_prot)
  elif name == 'scvi':
    model = InferenceSCVI(gene_dim=n_genes)
  elif name == 'scvinl':
    model = InferenceSCVI(gene_dim=n_genes, no_library_size=True)
  elif name == 'scvae':
    model = InferenceSCVAE(gene_dim=n_genes)
  elif name == 'dca':
    model = InferenceDCA(gene_dim=n_genes)
  else:
    raise NotImplementedError
  # All models are the same, fast and easy
  model.fit(X=X_train,
            y=y_train if model.is_semi_supervised else None,
            n_epoch=n_epoch, detail_logging=False)
  train_time = time.time() - start_time

  # ====== evaluation ====== #
  start_time = time.time()

  print("Start evaluating %s ..." % ctext(name, 'lightyellow'))
  pos = Posterior(model, ds=eval_ds)

  # chaining methods for convenience
  # first evaluate the latents
  pos.new_figure(nrow=12, ncol=12
  ).plot_latents_binary_scatter(test=False, ax=221, legend=False
  ).plot_latents_distance_heatmap(test=False, ax=222, legend=False
  ).plot_latents_binary_scatter(test=True, ax=223
  ).plot_latents_distance_heatmap(test=True, ax=224
  # ).plot_streamline_F1(mode='ovr'
  # ).plot_streamline_F1(mode='ovo'
  )
  # second evaluate the imputation
  pos.plot_correlation_marker_pairs(
  # ).plot_correlation_top_pairs(n=8, proteins=['CD4', 'CD8'], top=True
  # ).plot_correlation_top_pairs(n=8, proteins=['CD4', 'CD8'], top=False
  )

  # # third the protein prediction
  # pos.plot_protein_predicted_series(
  # ).plot_protein_scatter(
  # )
  # Finally debugging the training process
  pos.new_figure().plot_learning_curves()
  eval_time = time.time() - start_time

  # ====== save the results ====== #
  save_path = os.path.join(SAVE_FIGURE_PATH, name)
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  pos.save_plots(save_path)
  # pos.save_scores(SAVE_SCORES_PATH % name)

  # ====== save and return ====== #
  with open(SAVE_MODEL_PATH[name], 'wb') as f:
    pickle.dump(model, f)
  return name, len(pickle.dumps(model)), train_time, eval_time

# ====== training can be done using multiprocessing ====== #
with Pool(3) as p:
  for name, size, train_time, eval_time in p.map(
      train_and_return, all_models):
    print("Finish %s, model size: %.2f(kB), train: %.2f(sec), test: %.2f(sec)" %
      (ctext(name, 'lightyellow'), size / 1024, train_time, eval_time))

# ===========================================================================
# Comparison multiple systems
# ===========================================================================
all_posteriors = {name: Posterior(pickle.load(open(path, 'rb')),
                                  ds=eval_ds)
                  for name, path in SAVE_MODEL_PATH.items()}
# Create a combined analysis
analyzer = ResultsSheet(*list(all_posteriors.values()))

# same strategy, method chaining
analyzer.plot_correlation_marker_pairs(
).plot_latents_binary_scatter(pca=False
).plot_imputation_scatter(color_by_library=False
).plot_imputation_scatter(color_by_library=True
).plot_scores(score_type='pearson'
).plot_scores(score_type='cluster'
).plot_scores(score_type='classifier'
).save_plots(SAVE_COMPARE_FIGURE_PATH
).save_scores(SAVE_COMPARE_SCORES_PATH)

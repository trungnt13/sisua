from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from sisua.analysis import Posterior
from sisua.data import get_dataset, standardize_protein_name
from sisua.models import (SCVI, DeepCountAutoencoder, MultitaskVAE,
                          VariationalAutoEncoder)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Loading Data
# ===========================================================================
x, y = get_dataset('pbmc8kly')
print(x)
print(y)

x_train, x_test = x.split(train_percent=0.9)
y_train, y_test = y.split(train_percent=0.9)
x_train.assert_matching_cells(y_train)
x_test.assert_matching_cells(y_test)

n_genes = x.shape[1]
n_prots = y.shape[1]

# ===========================================================================
# Model configuration and training
# ===========================================================================
# The configurations for SISUA paper are:
# nlayers = 1
# hdim = 128
# zdim = 32
# However, it is recommended to keep this number as is if you run the code
# on Google Colab
nlayers = 1
hdim = 64
zdim = 16
epochs = 500

# ===========================================================================
# Create and train unsupervised model
# ===========================================================================
scvae = VariationalAutoEncoder(units=n_genes,
                               xdist='zinb',
                               nlayers=nlayers,
                               hdim=hdim,
                               zdim=zdim)
# Be generous with the number of epoch, since we use EarlyStopping,
# the algorithm will stop when overfitting
scvae.fit(x_train, epochs=epochs, verbose=True)

imputation, latent = scvae.predict(x_test)
# instead of getting a single imputation, we get a distribution of
# the imputation, which is much more helpful
print(imputation)
print(latent)

pos = Posterior(scvae, gene=x_test, protein=y_test, verbose=True)
# inspecting training process
pos.plot_learning_curves(metrics=['loss', 'klqp', 'loss_x', 'nllk_x'],
                         ignore_missing=True,
                         fig=(12, 4))

# the analysis of marker proteins (if available)
proteins = ['cd4', 'cd8']
pos.plot_correlation_marker_pairs(imputed=True, proteins=proteins, fig=(12, 10))
#
pos.plot_correlation_top_pairs(n=5, proteins=proteins, fig=(16, 6))
pos.plot_correlation_bottom_pairs(n=5, proteins=proteins, fig=(16, 6))

# the analysis of latent space
pos.plot_latents_binary_scatter()
pos.plot_latents_distance_heatmap()
pos.plot_latents_protein_pairs(figsize=(12, 4))
pos.plot_classifier_F1(x_train=x_train, y_train=y_train, figsize=(12, 12))
# all figure and analysis could be saved to pdf file for later inspectation
# pos.save_figures('/tmp/tmp.pdf')
# ===========================================================================
# The DeepCountAutoencoder could have deterministic or stochastic loss
# ===========================================================================
# deterministic loss
dca_detr = DeepCountAutoencoder(units=n_genes,
                                xdist='mse',
                                nlayers=nlayers,
                                hdim=hdim,
                                zdim=zdim)
dca_detr.fit(x_train, epochs=epochs, verbose=False)
imputation_dca1, latent_dca1 = dca_detr.predict(x_test)

# stochastic loss
dca_stch = DeepCountAutoencoder(units=n_genes,
                                xdist='zinb',
                                nlayers=nlayers,
                                hdim=hdim,
                                zdim=zdim)
dca_stch.fit(x_train, epochs=epochs, verbose=False)
imputation_dca2, latent_dca2 = dca_stch.predict(x_test)

# both model return a distribution, which is well generalized by the SISUA
# framework, however, there is important different when you sampling from
# imputation distribution
print(imputation_dca1)
print(imputation_dca2)

# sample and plot the differences
n = 4
sample1 = imputation_dca1.sample(n).numpy()
sample2 = imputation_dca2.sample(n).numpy()
labels = np.argmax(y_test.probabilistic_embedding().obsm['X_prob'], axis=1)
labels_name = [standardize_protein_name(i) for i in y_test.var['protid']]
labels = [labels_name[i] for i in labels]

plt.figure(figsize=(15, 6))
for fig_id, (model_name, sample) in enumerate([("DCA-deterministic", sample1),
                                               ("DCA-stochastic", sample2)]):
  x = []
  y = []
  index = []
  label = []
  for i, s in enumerate(sample):
    s = np.squeeze(s, axis=0)

    pca = PCA(n_components=2, random_state=8).fit_transform(s)
    x.append(pca[:, 0])
    y.append(pca[:, 1])

    index.append(np.full(shape=(s.shape[0],), fill_value=i + 1, dtype='int32'))

    label.append(labels)

  x = np.concatenate(x, axis=0)
  y = np.concatenate(y, axis=0)
  index = np.concatenate(index, axis=0)
  label = np.concatenate(label, axis=0)

  ax = plt.subplot(1, 2, fig_id + 1)
  sns.scatterplot(x='x',
                  y='y',
                  hue='label',
                  style='sample',
                  alpha=0.5,
                  data={
                      'x': x,
                      'y': y,
                      'label': label,
                      'sample': index
                  },
                  ax=ax)
  ax.set_title(model_name)
# ===========================================================================
# Two latents are returned for scVI
# ===========================================================================
# Another parameterization of Negative Binomial distribution must be used
# for scVI, i.e. Negative Binomial with mean and 'D'ispersion parameters
scvi = SCVI(units=n_genes, xdist='zinbd', nlayers=nlayers, hdim=hdim, zdim=zdim)
scvi.fit(x_train, epochs=epochs, verbose=False)
imputation, (latent, log_library) = scvi.predict(x_test)
print(imputation)
print(latent)
print(log_library)

# we check how efficient scVI modeling the library size
lib = np.sum(x_test.X, axis=1)
indices = np.argsort(lib)

# Original data is the continuous line
# Sample from the model is the dashed lines
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(lib[indices], label="Original Library")
ax = ax.twinx()
for i in range(4):
  # what is get is the log library
  s = np.exp(log_library.sample().numpy().ravel())
  ax.plot(s, linestyle='--', label="Sample#%d" % (i + 1))
ax.legend()

# ===========================================================================
# Create and train semi-supervised model
# ===========================================================================
sisua = MultitaskVAE(units=[n_genes, n_prots],
                     xdist=['zinb', 'nb'],
                     nlayers=nlayers,
                     hdim=hdim,
                     zdim=zdim)
# Be generous with the number of epoch, since we use EarlyStopping,
# the algorithm will stop when overfitting
sisua.fit([x_train, y_train], epochs=epochs, verbose=True)

(imputation, protein), latent = sisua.predict(x_test)
print(imputation)
print(protein)
print(latent)

# only SISUA have the ability to make analysis for predicted protein levels
pos = Posterior(sisua, gene=x_test, protein=y_test, verbose=True)
pos.plot_protein_predicted_series(proteins=['cd4', 'cd8'])
pos.plot_protein_scatter(protein_name='CD4')
pos.plot_protein_scatter(protein_name='CD8')
# pos.save_figures()

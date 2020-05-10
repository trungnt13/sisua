from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.backend import interpolation
from odin.ml import fast_pca, fast_tsne, fast_umap
from sisua.data import get_dataset, standardize_protein_name
from sisua.label_threshold import ProbabilisticEmbedding
from sisua.models import (SCALE, SCVI, SISUA, DeepCountAutoencoder,
                          NetworkConfig, RandomVariable, VariationalAutoEncoder)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

sns.set()
# ===========================================================================
# Configuration
# ===========================================================================
SAVE_PATH = '/tmp/uncertainty'
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

# or fast_pca, or fast_umap, or fast_tsne
fn_dim_reduction = partial(fast_umap, n_components=2)
train_config = dict(batch_size=64, epochs=200, verbose=True)
sample_shape = 2
latent_dim = 10

network = NetworkConfig(use_conv=True, pyramid=True, conv_proj=128)
kl = interpolation.const(vmax=1)
# kl = interpolation.linear(vmin=0,
#                           vmax=10,
#                           norm=20,
#                           cyclical=True,
#                           delayOut=5,
#                           delayIn=5)
# maximum amount of data points for testing (visualization)
n_samples_visualization = 300
DS_NAME = 'pbmc8kly'
# ===========================================================================
# Load data
# ===========================================================================
gene, prot = get_dataset(DS_NAME)
X_train, X_test = gene.split()
y_train, y_test = prot.split()
print("Labels:", prot.var)

gene_rv = RandomVariable(gene.n_vars, posterior='zinbd', name='rna')
prot_rv = RandomVariable(prot.n_vars, posterior='nb', name='adt')

# ====== prepare the labels ====== #
labels_name = standardize_protein_name(prot.var.iloc[:, 0].to_numpy())
if not y_test.is_binary:
  y_test.probabilistic_embedding()
  labels = np.argmax(y_test.obsm['X_prob'], axis=-1)
else:
  labels = np.argmax(y_test.X, axis=-1)
labels = np.array([labels_name[i] for i in labels])

# ====== downsample for visualization ====== #
if len(X_test) > n_samples_visualization:
  ids = np.random.permutation(len(X_test))[:n_samples_visualization]
  X_test = X_test.X[ids]
  labels = labels[ids]
# ===========================================================================
# Create the inference
# ===========================================================================
N_MODELS = 4
# scVI
scvi = SCVI(outputs=gene_rv,
            latent_dim=latent_dim,
            network=network,
            kl_interpolate=kl)
scvi.fit(X_train, **train_config)
scvi.plot_learning_curves().save_figures(os.path.join(SAVE_PATH, 'scvi.pdf'))

# scale
scale = SCALE(outputs=gene_rv,
              latent_dim=latent_dim,
              network=network,
              kl_interpolate=kl)
scale.fit(X_train, **train_config)
scale.plot_learning_curves().save_figures(os.path.join(SAVE_PATH, 'scale.pdf'))

# scVAE
scvae = VariationalAutoEncoder(outputs=gene_rv,
                               latent_dim=latent_dim,
                               network=network,
                               kl_interpolate=kl)
scvae.fit(X_train, **train_config)
scvae.plot_learning_curves().save_figures(os.path.join(SAVE_PATH, 'scvae.pdf'))

# SISUA
sisua = SISUA(rna_dim=gene.n_vars,
              adt_dim=prot.n_vars,
              latent_dim=latent_dim,
              network=network,
              kl_interpolate=kl)
sisua.fit([X_train, y_train], **train_config)
sisua.plot_learning_curves().save_figures(os.path.join(SAVE_PATH, 'sisua.pdf'))


# ===========================================================================
# Sampling
# ===========================================================================
# sample API for all models (Inferences)
def create_sample(model):
  pX, pZ = model.predict(X_test, apply_corruption=False, verbose=False)
  if isinstance(pZ, tuple):
    pZ = pZ[0]
  mean = pZ.mean()
  samples = pZ.sample(sample_shape)
  return mean, samples

scvi = create_sample(model=scvi)
scale = create_sample(model=scale)
scvae = create_sample(model=scvae)
sisua = create_sample(model=sisua)

# ===========================================================================
# Visualization
# ===========================================================================
# config for the mean
fig_config = dict(size=66, grid=False, color=labels, alpha=1.0, linewidths=0.)
# config for the samples
fig_config1 = dict(fig_config)
fig_config1['size'] = 20
fig_config1['alpha'] = 0.5
fig_config1['linewidths'] = 1.0
fig_config1['marker'] = 'x'

vs.plot_figure(nrow=12, ncol=20)

# ====== helper plotting ====== #
def plot_latent(data, idx, title):
  mean, samples = data
  # only the mean
  ax = vs.subplot(2, N_MODELS, idx)
  vs.plot_scatter(fn_dim_reduction(mean),
                  ax=ax,
                  title='[Only Mean]' + title,
                  legend_enable=False,
                  **fig_config)
  # mean and sample (single t-SNE)
  ax = vs.subplot(2, N_MODELS, idx + N_MODELS)
  z = np.concatenate([np.expand_dims(mean, axis=0), samples], axis=0)
  z = np.reshape(fn_dim_reduction(z.reshape(-1, z.shape[-1])),
                 z.shape[:-1] + (2,))
  for i in z[1:]:
    vs.plot_scatter(i, ax=ax, legend_enable=False, **fig_config1)
  vs.plot_scatter(z[0],
                  ax=ax,
                  title='[Both Mean and Samples]' + title,
                  legend_enable=True,
                  **fig_config)

# ====== plot ====== #
plot_latent(data=scvi, idx=1, title='scVI')
plot_latent(data=scvae, idx=2, title='scVAE')
plot_latent(data=scale, idx=3, title='SCALE')
plot_latent(data=sisua, idx=4, title='SISUA')
plt.suptitle(DS_NAME)

# ====== save the figure ====== #
vs.plot_save(os.path.join(SAVE_PATH, 'compare.pdf'), log=True)

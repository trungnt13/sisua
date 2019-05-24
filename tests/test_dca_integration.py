# DCA version 0.2.2
# This script fit 2 identical DCA models on multiple datasets to validate
# the DCA integration within SISUA.
from __future__ import print_function, division, absolute_import

import numpy as np

from odin.backend import log_norm

from sisua.data import (Cortex, PBMCscVI, PBMCscVAE,
                        apply_artificial_corruption)
from sisua.inference import InferenceDCA
from sisua.inference.inference_dca import dca_normalize

from dca.api import dca, normalize

# ===========================================================================
# Configuration
# ===========================================================================
SAVE_FIGURE_PATH = '/tmp/tmp.pdf'

SEED = 5218
n_epoch = 200
batch_size = 128
LEARNING_RATE = 1e-4

BATCH_SIZE = 128

n_hidden = 128
n_latent = 32

dropout_rate = 0.1

corruption_rate = 0.25
corruption_dist = 'binomial'

# ===========================================================================
# Loading the dataset
# ===========================================================================
# ds = Cortex()
ds = PBMCscVI()
print(ds)

X = ds['X'][:]
y = ds['y'][:]
labels = ds['y_col'][:]
y = [labels[i] for i in np.argmax(y, axis=-1)]

X_train = apply_artificial_corruption(X,
  dropout=corruption_rate, distribution=corruption_dist)

n_genes = X_train.shape[1]
# ===========================================================================
# Training DCA
# ===========================================================================
from anndata import AnnData
tmp = AnnData(X_train)
dca_model = dca(
    tmp,
    mode='denoise',
    ae_type='zinb-conddisp',
    normalize_per_cell=True,
    scale=True,
    log1p=True,
    hidden_size=(n_hidden, n_latent, n_hidden), # network args
    hidden_dropout=0.,
    batchnorm=True,
    activation='relu',
    init='glorot_uniform',
    epochs=n_epoch,               # training args
    reduce_lr=10,
    early_stop=15,
    batch_size=batch_size,
    optimizer='adam',
    random_state=0,
    threads=None, verbose=True,
    network_kwds={'input_dropout': dropout_rate},
    training_kwds={'learning_rate': LEARNING_RATE},
    return_model=True,
    return_info=False,
    copy=False
)
dca_history = dca_model.model.history.history
print(type(dca_model))
dca_outputs = dca_model.predict(tmp, 'full')
dca_Z = dca_outputs['reduced']
dca_V = dca_outputs['mean']
# ===========================================================================
# Training SISUA-DCA
# ===========================================================================
sisua_model = InferenceDCA(gene_dim=n_genes,
               dispersion='gene-cell',
               xnorm='log',
               xdist='zinb', ydist='bernoulli', zdist='normal',
               xdrop=dropout_rate,
               hdim=n_hidden, zdim=n_latent, nlayer=1)
sisua_model.fit(X_train, validation_percent=0.1, n_mcmc_samples=1,
          corruption_rate=0.,
          batch_size=batch_size, n_epoch=n_epoch,
          learning_rate=LEARNING_RATE,
          detail_logging=True)
sisua_history = sisua_model._ae.model.history.history
sisua_Z = sisua_model.predict_Z(X_train)
sisua_V = sisua_model.predict_V(X_train)
# ===========================================================================
# Now some testing
# ===========================================================================
assert dca_Z.shape == sisua_Z.shape
assert dca_V.shape == sisua_V.shape

from matplotlib import pyplot as plt
from odin import visual as V
import seaborn as sns

from odin.ml import fast_tsne, fast_pca
from sisua.analysis.latent_benchmarks import (
    streamline_classifier, clustering_scores)

# ====== training history ====== #
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sisua_history['loss'], label="Train")
plt.plot(sisua_history['val_loss'], label="Val")
plt.legend()
plt.grid(True)
plt.title("SISUA")

plt.subplot(1, 2, 2)
plt.plot(dca_history['loss'], label='Train')
plt.plot(dca_history['val_loss'], label='Val')
plt.legend()
plt.grid(True)
plt.title("DCA")

# ====== the latent space ====== #
tmp = np.argmax(ds['y'][:], axis=-1)
sisua_scores = clustering_scores(latent=sisua_Z, labels=tmp, n_labels=len(labels))
sisua_scores = '\n' + ','.join(['%s:%.2f' % (k, v)
                                for k, v in sorted(sisua_scores.items())])
dca_scores = clustering_scores(latent=dca_Z, labels=tmp, n_labels=len(labels))
dca_scores = '\n' + ','.join(['%s:%.2f' % (k, v)
                              for k, v in sorted(dca_scores.items())])

streamline_classifier(Z_train=sisua_Z, y_train=tmp, Z_test=sisua_Z, y_test=tmp,
                      labels_name=labels, title="SISUA")
streamline_classifier(Z_train=dca_Z, y_train=tmp, Z_test=dca_Z, y_test=tmp,
                      labels_name=labels, title="DCA")

sisua_Z = fast_tsne(sisua_Z, random_state=SEED)
dca_Z = fast_tsne(dca_Z, random_state=SEED)

plt.figure(figsize=(12, 6))
V.plot_scatter(x=dca_Z, color=y, legend_enable=True, title='SISUA' + sisua_scores, ax=121, size=8)
V.plot_scatter(x=sisua_Z, color=y, legend_enable=True, title='DCA' + dca_scores, ax=122, size=8)
# ====== the denoised gene expression ====== #
L_org = log_norm(np.sum(X, -1), axis=0)
L_sisua = log_norm(np.sum(sisua_V, -1), axis=0)
L_dca = log_norm(np.sum(dca_V, -1), axis=0)

pca, _ = fast_pca(X, n_components=2, return_model=True, random_state=SEED)
org = pca.transform(X)
dca = pca.transform(dca_V)
sisua = pca.transform(sisua_V)

plt.figure(figsize=(18, 6))
V.plot_scatter(x=org, color=y, legend_enable=True, title='ORG', ax=131, size=8)
V.plot_scatter(x=sisua, color=y, legend_enable=True, title='SISUA', ax=132, size=8)
V.plot_scatter(x=dca, color=y, legend_enable=True, title='DCA', ax=133, size=8)

plt.figure(figsize=(18, 6))
V.plot_scatter_heatmap(x=org, val=L_org, legend_enable=True, title='ORG', ax=131, size=8)
V.plot_scatter_heatmap(x=sisua, val=L_sisua, legend_enable=True, title='SISUA', ax=132, size=8)
V.plot_scatter_heatmap(x=dca, val=L_dca, legend_enable=True, title='DCA', ax=133, size=8)

# ====== save everything ====== #
V.plot_save(SAVE_FIGURE_PATH)

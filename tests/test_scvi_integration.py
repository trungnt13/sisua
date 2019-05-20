# scVI version 0.2.4
# This script fit 2 identical scVI models on Cortex dataset to validate
# if the `InferenceSCVI` and original scVI model matching the performance
# https://github.com/YosefLab/scVI/blob/master/tests/notebooks/scVI_reproducibility.ipynb
from __future__ import print_function, division, absolute_import

import torch
import numpy as np

from odin.stats import summary
from odin.utils import ctext, UnitTimer, one_hot

from scvi.dataset import (CortexDataset, RetinaDataset, CiteSeqDataset,
                          SyntheticRandomDataset, PbmcDataset,
                          HematoDataset, RetinaDataset)
from scvi.models import *
from scvi.inference import UnsupervisedTrainer, SemiSupervisedTrainer

from sisua.inference import InferenceSCVI

# ===========================================================================
# Configuration
# ===========================================================================
SAVE_DATA_PATH = '/tmp/bio_data'
SAVE_FIGURE_PATH = '/tmp/tmp.pdf'

n_epoch = 250
LEARNING_RATE = 1e-4
# LEARNING_RATE = 0.001

TRAIN_SIZE = 0.8
BATCH_SIZE = 128

n_hidden = 128
n_latent = 32
n_layer = 2

dropout_rate = 0.1
# dispersion = 'gene-cell'
dispersion = 'gene'

# corruption_dist = 'uniform'
# corruption_rate = 0.1
corruption_dist = 'binomial'
corruption_rate = 0.25

log_variational = True

# ====== others ====== #
torch_device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
  """ numpy array to pytorch tensor """
  return torch.from_numpy(x.astype('float32')).to(torch_device)

def to_array(x):
  """ pytorch tensor to numpy array """
  if hasattr(x, 'todense'):
    return np.array(x.todense())
  if hasattr(x, 'cpu'):
    return x.data.cpu().numpy()
  return x

# Load dataset
cortex = CortexDataset(save_path=SAVE_DATA_PATH)
X = cortex.X
labels = cortex.cell_types
n_labels = len(labels)
Y = one_hot(cortex.labels.ravel(), n_labels)

# ===========================================================================
# scVI
# ===========================================================================
scvi = VAE(n_input=cortex.nb_genes, n_batch=0, n_labels=0,
           n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layer,
           dispersion=dispersion, dropout_rate=dropout_rate,
           log_variational=log_variational)
trainer = UnsupervisedTrainer(
    model=scvi, gene_dataset=cortex,
    use_cuda=torch.cuda.is_available(),
    train_size=TRAIN_SIZE, frequency=1, benchmark=False,
    data_loader_kwargs=dict(batch_size=BATCH_SIZE),
    metrics_to_monitor=None, verbose=False)
trainer.corrupt_posteriors(rate=corruption_rate,
                           corruption=corruption_dist)
trainer.train(n_epochs=n_epoch, lr=LEARNING_RATE)
trainer.uncorrupt_posteriors()
# ignore epoch 0 which only depend on initialization process
scvi_loss = (trainer.history["ll_train_set"][1:],
             trainer.history["ll_test_set"][1:])

# ===========================================================================
# SISUA
# ===========================================================================
sisua = InferenceSCVI(gene_dim=cortex.nb_genes, prot_dim=None,
               dispersion=dispersion,
               xnorm='log' if log_variational else 'raw',
               xclip=0, yclip=0,
               xdist='zinb',
               xdrop=dropout_rate,
               hdim=n_hidden, zdim=n_latent, nlayer=n_layer)
sisua.fit(X=cortex.X, validation_percent=1 - TRAIN_SIZE,
          n_mcmc_samples=1,
          corruption_rate=corruption_rate, corruption_dist=corruption_dist,
          batch_size=BATCH_SIZE, n_epoch=n_epoch,
          learning_rate=LEARNING_RATE,
          detail_logging=False)
sisua_loss = (sisua.history['train']['loss'], sisua.history['valid']['loss'])

# ===========================================================================
# Get all outputs
# ===========================================================================
np.random.seed(5218)
ids = np.random.permutation(len(X))
n_train = len(X) - 300
train_ids = ids[:n_train]
test_ids = ids[n_train:(n_train + 300)]
x = to_tensor(X[train_ids])
y = Y[train_ids]
library_size = np.sum(X[train_ids], axis=-1)

x_test = to_tensor(X[test_ids])
y_test = Y[test_ids]

# px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library
scvi_outputs = scvi.inference(x)
scvi_outputs = [to_array(i) for i in scvi_outputs]

sisua_outputs = sisua._vae.inference(x)
sisua_outputs = [to_array(i) for i in sisua_outputs]

if log_variational:
  x = torch.log(1 + x)
scvi_z = to_array(scvi.z_encoder(x)[0].cpu())
sisua_z = to_array(sisua._vae.z_encoder(x)[0].cpu())

scvi_ztest = to_array(scvi.z_encoder(x_test)[0].cpu())
sisua_ztest = to_array(sisua._vae.z_encoder(x_test)[0].cpu())

for i, j in zip(scvi_outputs, sisua_outputs):
  assert i.shape == j.shape
assert scvi_z.shape == sisua_z.shape
assert scvi_ztest.shape == sisua_ztest.shape

# ===========================================================================
# Comparison
# ===========================================================================
import seaborn as sns
from odin import visual as V
from matplotlib import pyplot as plt
from sisua.analysis.latent_benchmarks import (
    plot_latents_binary, clustering_scores, streamline_classifier)
from sisua.analysis.imputation_benchmarks import (
    imputation_score, imputation_mean_score, imputation_std_score,
    plot_imputation)

# ====== training process ====== #
V.plot_figure(nrow=4, ncol=10)
plt.subplot(1, 2, 1)
plt.plot(scvi_loss[0], label='train')
plt.plot(scvi_loss[1], label='valid')
plt.legend()
plt.title('scVI')

plt.subplot(1, 2, 2)
plt.plot(sisua_loss[0], label='train')
plt.plot(sisua_loss[1], label='valid')
plt.legend()
plt.title('SISUA')
plt.tight_layout()

# ====== Latent space ====== #
V.plot_figure(nrow=8, ncol=18)
plot_latents_binary(Z=scvi_z, y=y, labels_name=labels, title="scVI",
             use_PCA=False, show_legend=True, ax=(1, 2, 1),
             enable_separated=False)
plot_latents_binary(Z=sisua_z, y=y, labels_name=labels, title="SISUA",
             use_PCA=False, show_legend=True, ax=(1, 2, 2),
             enable_separated=False)

# # ====== classifier results ====== #
streamline_classifier(Z_train=scvi_z, y_train=y,
                      Z_test=scvi_ztest, y_test=y_test,
                      labels_name=labels, title='scVI')
streamline_classifier(Z_train=sisua_z, y_train=y,
                      Z_test=sisua_ztest, y_test=y_test,
                      labels_name=labels, title='SISUA')

# ====== imputation ====== #
V.plot_figure(nrow=6, ncol=18)
ids = np.argsort(library_size)
plt.plot(library_size[ids], label="Original", linewidth=2.0, linestyle='--')
plt.plot(scvi_outputs[-3].ravel()[ids], label="scVI", linewidth=2.0)
plt.plot(sisua_outputs[-3].ravel()[ids], label="SISUA", linewidth=2.0)
plt.legend()
plt.title("Library Size")

x = to_array(x)
scvi_score = imputation_score(original=x, imputed=scvi_outputs[2])
sisua_score = imputation_score(original=x, imputed=sisua_outputs[2])
print("scVI:", scvi_score)
print("SISUA:", scvi_score)

# ====== save all the figure ====== #
V.plot_save(SAVE_FIGURE_PATH, dpi=48)

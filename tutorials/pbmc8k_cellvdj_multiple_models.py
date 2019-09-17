from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import tensorflow as tf
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
from sklearn.mixture import GaussianMixture
from tensorflow_probability.python.layers.distribution_layer import (
    IndependentNormal, MixtureNormal, MixtureSameFamily)

from odin.bay.distributions import Independent, ZeroInflated
from odin.networks import MixtureDensityNetwork
from sisua.data import SingleCellOMIC, get_dataset, standardize_protein_name
from sisua.models import (SCVI, DeepCountAutoencoder, MultitaskVAE, MultitaskVI,
                          SCScope, VariationalAutoEncoder)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.random.set_seed(8)
np.random.seed(8)

# ===========================================================================
# Configuration
# ===========================================================================
# ====== configs ====== #
epochs = 500
batch_size = 64
batch_size_pred = 16
n_mcmc_prediction = 3
VDJ_percent = 0.1  # 0.2 = ~10,000 cells

# ====== flags ====== #
enable_training = True
pbmc8k_predict = True
cellvdj_predict = True
override = True

# ====== save path for all trained models ====== #
path = '/tmp/pbmc8k_cellvdj'
if os.path.exists(path) and override:
  print("Overriding path: %s" % path)
  shutil.rmtree(path)
if not os.path.exists(path):
  os.mkdir(path)
if os.path.isfile(path):
  raise ValueError("'%s' must be folder path" % path)

# ===========================================================================
# Train on PBMC8k-ly
# ===========================================================================
x, y = get_dataset('pbmc8kly')
x_train, x_test = x.split()
y_train, y_test = y.split()
gene_name1 = x.var['geneid']

n_genes = x.shape[1]
n_prot = y.shape[1]

all_models = [
    DeepCountAutoencoder(units=n_genes),
    SCVI(units=n_genes),
    VariationalAutoEncoder(units=n_genes),
    MultitaskVAE(units=[n_genes, n_prot]),
    MultitaskVI(units=[n_genes, n_prot]),
]

# ====== training all the models ====== #
if enable_training:
  kw = dict(epochs=epochs, batch_size=batch_size)
  for m in all_models:
    print("\nTraining model: %s" % m.id)
    if m.is_semi_supervised:
      m.fit([x_train, y_train], **kw)
    else:
      m.fit(x_train, **kw)

# ====== Generate prediction for PBMC8k ====== #
all_8k_gene = []
all_8k_prot = []
all_8k_latents = []
if pbmc8k_predict:
  for m in all_models:
    outputs, latents = m.predict(x_test,
                                 n_samples=n_mcmc_prediction,
                                 batch_size=batch_size_pred,
                                 apply_corruption=False,
                                 enable_cache=False)

    if isinstance(latents, (tuple, list)):
      latents = latents[0]
    all_8k_latents.append(latents)

    if m.is_semi_supervised:
      all_8k_gene.append(outputs[0])
      all_8k_prot.append(outputs[1])
    else:
      all_8k_gene.append(outputs)
      all_8k_prot.append(None)

# ===========================================================================
# Make prediction for CellVDJ
# ===========================================================================
x, y = get_dataset('cellvdj')
# matching the Gene set of pbmc8kly and cellvdj
gene_name2 = {name: idx for idx, name in enumerate(x.var['genename'])}
gene_ids = [gene_name2[name] for name in gene_name1 if name in gene_name2]
new_X = x.X[:, gene_ids]
new_var = x.var.iloc[gene_ids]
assert np.all(new_var['genename'].values == gene_name1.values)
# create new gene dataset
x = SingleCellOMIC(X=new_X, obs=x.obs, var=new_var)
# too big data just get a percent of Cells
x, _ = x.split(train_percent=VDJ_percent)
y, _ = y.split(train_percent=VDJ_percent)

# ====== generate the prediction for CellVDJ ====== #
all_vdj_gene = []
all_vdj_prot = []
all_vdj_latents = []
if cellvdj_predict:
  for m in all_models:
    outputs, latents = m.predict(x,
                                 n_samples=n_mcmc_prediction,
                                 batch_size=batch_size_pred,
                                 apply_corruption=False,
                                 enable_cache=False)
    if isinstance(latents, (tuple, list)):
      latents = latents[0]
    all_vdj_latents.append(latents)

    if m.is_semi_supervised:
      all_vdj_gene.append(outputs[0])
      all_vdj_prot.append(outputs[1])
    else:
      all_vdj_gene.append(outputs)
      all_vdj_prot.append(None)

# ===========================================================================
# Saving everything
# ===========================================================================
np.savetxt(os.path.join(path, 'cellid_8k.txt'),
           x_test.obs['cellid'].values,
           fmt='%s')
np.savetxt(os.path.join(path, 'cellid_vdj.txt'),
           x.obs['cellid'].values,
           fmt='%s')

np.savetxt(os.path.join(path, 'gene_name.txt'),
           x.var['genename'].values,
           fmt='%s')

np.savetxt(os.path.join(path, 'protein_8k.txt'),
           [standardize_protein_name(i) for i in y_test.var['protid']],
           fmt='%s')
np.savetxt(os.path.join(path, 'protein_vdj.txt'),
           y.var['protid'].values,
           fmt='%s')

mmwrite(os.path.join(path, 'x_8k.mtx'), coo_matrix(x_test.X, dtype='float32'))
mmwrite(os.path.join(path, 'y_8k.mtx'), y_test.X)

mmwrite(os.path.join(path, 'x_vdj.mtx'), coo_matrix(x.X, dtype='float32'))
mmwrite(os.path.join(path, 'y_vdj.mtx'), y.X)


# ====== helper function ====== #
def save_data(arrays, prefix, dataset):
  for model, dist in zip(all_models, arrays):
    name = model.id
    if dist is None:
      continue
    if isinstance(dist, Independent) and isinstance(dist.distribution,
                                                    ZeroInflated):
      dist = dist.distribution.count_distribution
    mean = dist.mean().numpy()  # only save the mean
    if mean.ndim == 3:
      mean = np.mean(mean, axis=0)
    outpath = os.path.join(path, '%s_%s_%s' % (prefix, name, dataset))
    print("Write data %s %s to path: %s" % (name, mean.shape, outpath))
    mmwrite(outpath, mean.astype('float32'))


if pbmc8k_predict:
  save_data(all_8k_gene, prefix='ximpu', dataset='8k')
  save_data(all_8k_prot, prefix='ypred', dataset='8k')
  save_data(all_8k_latents, prefix='z', dataset='8k')

if cellvdj_predict:
  save_data(all_vdj_gene, prefix='ximpu', dataset='vdj')
  save_data(all_vdj_prot, prefix='ypred', dataset='vdj')
  save_data(all_vdj_latents, prefix='z', dataset='vdj')

with open(os.path.join(path, "README.txt"), 'w') as f:
  f.write("""
Dataset:
8k : the test set of PBMC8k-ly
vdj : about > 5000 cells (10%) of CellVDJ dataset

Annotation:
x : original gene expression
y : orginal protein levels

ximpu : imputed gene expression
ypred : predicted protein levels
z : latent space

Model:
dca : Deep Count Autoencoder
scvi : scVI - Single-cell Variation Inference
vae : scVAE - single-cell Variational Autoencoder
mvae : SISUA - semi-supervised version of scVAE
mvi : semi-supervised version of scVI
""")

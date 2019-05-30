from __future__ import print_function, division, absolute_import

import numpy as np

from odin import visual as vs
from odin.ml import fast_tsne, fast_pca

from sisua.data import get_dataset
from sisua.inference import InferenceSISUA, InferenceSCVI, InferenceSCVAE

# ===========================================================================
# Configuration
# ===========================================================================
SAVE_FIGURE_PATH = '/tmp/tmp.pdf'
fn_dim_reduction = fast_tsne # or fast_pca

train_config = dict(
    batch_size=66,
    n_epoch=200,
    detail_logging=True,
)
n_samples = 4
# ===========================================================================
# Load data
# ===========================================================================
ds, gene, prot = get_dataset('pbmcscvi')
print(ds)

X_train = gene.get_data('train')
X_test = gene.get_data('test')

y_train = prot.get_data('train')
y_test = prot.get_data('test')

labels = ds['y_col']
print("Labels:", ', '.join(labels))
y_test = np.array([labels[int(i)]
                   for i in np.argmax(y_test, axis=1)])
# ===========================================================================
# Create the inference
# ===========================================================================
# scVI
scvi = InferenceSCVI(gene_dim=gene.feat_dim)
scvi.fit(X_train, **train_config)

# scVAE
scvae = InferenceSCVAE(gene_dim=gene.feat_dim)
scvae.fit(X_train, **train_config)

# SISUA
sisua = InferenceSISUA(gene_dim=gene.feat_dim, prot_dim=prot.feat_dim)
sisua.fit(X_train, y_train, **train_config)

# ===========================================================================
# Sampling
# ===========================================================================
# sample API for all models (Inferences)
def create_sample(model):
  z = dict(
      z_mean=model.predict_Z(X_test),
      z_sample=model.sample_Z(X_test, n_samples),
  )
  return z

scvi = create_sample(model=scvi)
scvae = create_sample(model=scvae)
sisua = create_sample(model=sisua)

# ===========================================================================
# Visualization
# ===========================================================================
# config for the mean
fig_config = dict(
    size=88,
    grid=False,
    color=y_test,
    alpha=1.0,
    linewidths=0.
)
# config for the samples
fig_config1 = dict(fig_config)
fig_config1['size'] = 20
fig_config1['alpha'] = 0.4
fig_config1['linewidths'] = 1.0
fig_config1['marker'] = 'x'

vs.plot_figure(nrow=12, ncol=18)

# ====== helper plotting ====== #
def plot_latent(data, idx, title):
  z_mean = fn_dim_reduction(data['z_mean'])

  # only the mean
  ax = vs.subplot(2, 3, idx)
  vs.plot_scatter(z_mean,
                  ax=ax, title='[Only Mean]' + title,
                  legend_enable=False, **fig_config)

  # mean and sample (single t-SNE)
  ax = vs.subplot(2, 3, idx + 3)
  z = np.concatenate((np.expand_dims(data['z_mean'], axis=0),
                      data['z_sample']), axis=0)
  z = np.reshape(fast_tsne(z.reshape(-1, z.shape[-1])),
                 z.shape[:-1] + (2,))
  for i in z[1:]:
    vs.plot_scatter(i, ax=ax,
                    legend_enable=False, **fig_config1)
  vs.plot_scatter(z[0],
                  ax=ax, title='[Both Mean and Samples]' + title,
                  legend_enable=True, **fig_config)

# ====== plot ====== #
plot_latent(data=scvi, idx=1, title='scVI')
plot_latent(data=scvae, idx=2, title='scVAE')
plot_latent(data=sisua, idx=3, title='SISUA')

# ====== save the figure ====== #
vs.plot_save(SAVE_FIGURE_PATH, log=True)

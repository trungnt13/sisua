from __future__ import absolute_import, division, print_function

import os
import shutil

from odin.utils import stdio
from sisua.analysis.sc_metrics import CorrelationScores, NegativeLogLikelihood
from sisua.data import get_dataset
from sisua.models.autoencoder import DeepCountAutoencoder
from sisua.models.scvi_models import SCVI
from sisua.models.semi_supervised import MultitaskVAE
from sisua.models.variational_autoencoder import VariationalAutoEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

epochs = 200
batch_size = 128
max_evals = 80
algorithm = 'bayes'
freq = 1000  # mean that only run on_train_end

path = '/tmp/autotune'
if os.path.exists(path):
  shutil.rmtree(path)
os.mkdir(path)

# sc_metrics more robust to NaN values
# TODO: accept a list of loss_name
stdio(os.path.join(path, 'fit_hyper.txt'))
# ===========================================================================
# Cortext
# ===========================================================================
x, y = get_dataset('cortex')
x.filter_cells(min_counts=1).filter_genes(min_counts=1)
gene = x.shape[1]
prot = y.shape[1]

SCVI.fit_hyper(x,
               loss_name='nllk0',
               model_kwargs=dict(units=gene, xdist='zinbd'),
               fit_kwargs=dict(epochs=epochs,
                               batch_size=batch_size,
                               callbacks=[NegativeLogLikelihood(freq=freq)]),
               max_evals=max_evals,
               save_path=os.path.join(path, 'scvi_cortex'),
               algorithm=algorithm,
               verbose=True)

DeepCountAutoencoder.fit_hyper(
    x,
    loss_name='nllk0',
    model_kwargs=dict(units=gene, xdist='zinbd'),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[NegativeLogLikelihood(freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'dca_cortex'),
    algorithm=algorithm,
    verbose=True)

VariationalAutoEncoder.fit_hyper(
    x,
    loss_name='nllk0',
    model_kwargs=dict(units=gene, xdist='zinbd'),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[NegativeLogLikelihood(freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'scvae_cortex'),
    algorithm=algorithm,
    verbose=True)

MultitaskVAE.fit_hyper(
    (x, y),
    loss_name='nllk0',
    model_kwargs=dict(units=(gene, prot), xdist=['zinbd', 'nbd']),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[NegativeLogLikelihood(freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'sisua_cortex'),
    algorithm=algorithm,
    verbose=True)

# ===========================================================================
# PBMC8k ly
# ===========================================================================
x, y = get_dataset('pbmc8kly')
gene = x.shape[1]
prot = y.shape[1]

SCVI.fit_hyper(x,
               loss_name='pearson_mean',
               model_kwargs=dict(units=gene, xdist='zinbd'),
               fit_kwargs=dict(
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=[CorrelationScores(extras=y, freq=freq)]),
               max_evals=max_evals,
               save_path=os.path.join(path, 'scvi_pbmc8k'),
               algorithm=algorithm,
               verbose=True)

DeepCountAutoencoder.fit_hyper(
    x,
    loss_name='pearson_mean',
    model_kwargs=dict(units=gene, xdist='zinbd'),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[CorrelationScores(extras=y, freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'dca_pbmc8k'),
    algorithm=algorithm,
    verbose=True)

VariationalAutoEncoder.fit_hyper(
    x,
    loss_name='pearson_mean',
    model_kwargs=dict(units=gene, xdist='zinbd'),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[CorrelationScores(extras=y, freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'scvae_pbmc8k'),
    algorithm=algorithm,
    verbose=True)

MultitaskVAE.fit_hyper(
    (x, y),
    loss_name='pearson_mean',
    model_kwargs=dict(units=(gene, prot), xdist=['zinbd', 'nbd']),
    fit_kwargs=dict(epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[CorrelationScores(extras=y, freq=freq)]),
    max_evals=max_evals,
    save_path=os.path.join(path, 'sisua_pbmc8k'),
    algorithm=algorithm,
    verbose=True)

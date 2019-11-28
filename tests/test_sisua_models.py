from __future__ import absolute_import, division, print_function

import os
import pickle
from enum import Flag, auto

import dill
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow_probability.python.distributions import Deterministic, Normal

from odin.bay.distribution_layers import DeterministicLayer, NormalLayer
from odin.networks import AdvanceModel, DenseDistribution, Parallel
# output, latent = ae.predict(x)
from sisua.analysis import Posterior
from sisua.data import get_dataset
from sisua.data.normalization_recipes import (CellRanger, Methods,
                                              NormalizationRecipe, Seurat,
                                              Sisua)
from sisua.models import get_model
from sisua.models.autoencoder import DeepCountAutoencoder
from sisua.models.semi_supervised import (MultiLatentVAE, MultitaskAutoEncoder,
                                          MultitaskVAE)
from sisua.models.variational_autoencoder import VariationalAutoEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(8)
np.random.seed(8)

x, y = get_dataset('pbmc8kly')
z = np.random.rand(*x.shape).astype('float32')
w = np.random.rand(*y.shape).astype('float32')
v = np.random.rand(*y.shape).astype('float32')
print(z.shape, w.shape)

for i, j in get_model().items():
  print(i, j)

x_train, x_test = x.split()
y_train, y_test = y.split()
print(np.sum(x_train.X))
print(np.sum(x_test.X))
assert np.all(x_train.obs['cellid'] == y_train.obs['cellid'])
assert np.all(x_test.obs['cellid'] == y_test.obs['cellid'])

#############
mae = MultitaskAutoEncoder()
mae.fit([x, y], epochs=8, n_mcmc=1)
mae.predict([x, y])

ae = DeepCountAutoencoder(loss='mse')
ae.fit(x, epochs=8, n_mcmc=1)
ae.predict(x)

ae = DeepCountAutoencoder(loss='poisson')
ae.fit(x, epochs=8, n_mcmc=1)
ae.predict(x)

ae = DeepCountAutoencoder(loss='nb')
ae.fit(x, epochs=8, n_mcmc=1)
ae.predict(x)

# DeepCountAutoencoder.fit_hyper(x, max_evals=20, fit_kwargs={'epochs': 2})
#############
vld = VariationalAutoEncoder(linear_decoder=True)
vld.fit(x, epochs=8, n_mcmc=1)
vld.predict(x)

vae = VariationalAutoEncoder()
vae.fit(x, epochs=8, n_mcmc=1)
vae.predict(x)

mvae = MultitaskVAE()
mvae.fit([x, y], epochs=8, n_mcmc=1)
mvae.predict([x, y])

mvld = MultitaskVAE(linear_decoder=True)
mvld.fit([x, y], epochs=8, n_mcmc=1)
mvld.predict([x, y])

from __future__ import absolute_import, division, print_function

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import EarlyStopping

from odin.bay import parse_distribution
from odin.networks import DenseDistribution, MixtureDensityNetwork
from sisua.data import get_dataset
from sisua.models.latents import NormalLatent
from sisua.models.networks import DenseNetwork

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(8)
np.random.seed(8)

x, y = get_dataset('pbmc8kly')
x_train, x_test = x.split()
y_train, y_test = y.split()
x_train.assert_matching_cells(y_train)
x_test.assert_matching_cells(y_test)


class MultitaskMDN(Model):

  def __init__(self, units, n_components=2):
    super(MultitaskMDN, self).__init__()
    self.encoder = DenseNetwork(name="Encoder", input_dropout=0.3)
    self.decoder = DenseNetwork(name="Decoder")
    self.latent = NormalLatent(units=32)
    self.output_gene = DenseDistribution(
        units=units[0],
        posterior=parse_distribution('zinb')[0],
        activation='linear',
        name='Gene')
    self.output_prot = MixtureDensityNetwork(units=units[1],
                                             n_components=n_components,
                                             covariance_type='none',
                                             name="Protein")

  def call(self, inputs, training=None, n_mcmc=1):
    x, y = inputs

    e = self.encoder(tf.math.log1p(x))
    qZ = self.latent(e)
    z = qZ.sample(n_mcmc)
    d = self.decoder(z)

    pX = self.output_gene(d)
    pY = self.output_prot(d)

    kl = self.latent.kl_divergence(analytic=True)
    llk_x = tf.expand_dims(pX.log_prob(x), axis=-1)
    llk_y = tf.convert_to_tensor(0, dtype=llk_x.dtype)
    llk_y = tf.expand_dims(pY.log_prob(y), axis=-1)

    elbo = llk_x + llk_y - kl
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)

    if training:
      self.add_loss(loss)

    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name='KLqp')
    self.add_metric(tf.reduce_mean(llk_x), aggregation='mean', name='llk_X')
    self.add_metric(tf.reduce_mean(llk_y), aggregation='mean', name='llk_Y')
    return x
    # return (pX, pY), qZ


# ===========================================================================
# Trainining and testing
# ===========================================================================
model = MultitaskMDN(units=(x.shape[1], y.shape[1]), n_components=2)
# ====== training ====== #
if True:
  model.compile(tf.optimizers.Adam(learning_rate=1e-4),
                experimental_run_tf_function=False)
  model.fit(x=(x_train.X, y_train.X),
            epochs=500,
            batch_size=64,
            verbose=True,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(min_delta=0.5,
                              patience=20,
                              verbose=True,
                              restore_best_weights=True)
            ])

  with open('/tmp/weights', 'wb') as f:
    pickle.dump(model.get_weights(), f)
  (pX, pY), qZ = model((x_test.X, y_test.X))
  print(pX)
  print(np.mean(pX.mean().numpy()))
  print(pY)
  print(np.mean(pY.mean().numpy()))
  print(qZ)
  print(np.mean(qZ.mean().numpy()))
# ====== testing ====== #
else:
  with open('/tmp/weights', 'rb') as f:
    weights = pickle.load(f)
  model.build(input_shape=[(None, x.shape[1]), (None, y.shape[1])])
  model.set_weights(weights)
  (pX, pY), qZ = model((x_test.X, y_test.X))
  print(pX)
  print(np.mean(pX.mean().numpy()))
  print(pY)
  print(np.mean(pY.mean().numpy()))
  print(qZ)
  print(np.mean(qZ.mean().numpy()))

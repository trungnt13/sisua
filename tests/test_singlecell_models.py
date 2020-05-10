from __future__ import absolute_import, division, print_function

import inspect
import os
import unittest

import numpy as np
import tensorflow as tf

from odin import bay
from sisua.data import OMIC, get_dataset
from sisua.models import (SCVI, SISUA, DeepCountAutoencoder, NetworkConfig,
                          RandomVariable, SingleCellModel,
                          VariationalAutoEncoder, get_all_models, get_model)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

_DS = "8kly"
_EPOCHS = 10


class ModelTest(unittest.TestCase):

  def _loss_not_rise(self, loss):
    loss = loss[1:]  # first epoch is messy
    is_decreasing = [i > j for i, j in zip(loss, loss[1:])]
    # 80% of the time, the training improving
    self.assertTrue(np.sum(is_decreasing) > 0.8 * (len(loss) - 1), str(loss))

  def test_find_all_models(self):
    for m in get_all_models():
      self.assertTrue(inspect.isclass(m) and issubclass(m, SingleCellModel))

  def test_network_config(self):
    # TODO

  def test_random_variable(self):
    x = np.random.rand(8, 12).astype(np.float32)

    rv = RandomVariable(dim=12, posterior='diag')
    dist = rv.create_posterior()
    y = dist(x)
    z = tf.convert_to_tensor(y)
    self.assertFalse(rv.is_zero_inflated)
    self.assertFalse(rv.is_deterministic)
    self.assertTrue(dist.event_shape == [12])
    self.assertTrue(isinstance(dist, bay.layers.DenseDistribution))
    self.assertTrue(
        isinstance(dist.posterior_layer, bay.layers.MultivariateNormalLayer))
    self.assertTrue(
        isinstance(dist.prior, bay.distributions.MultivariateNormalDiag))
    self.assertTrue(isinstance(y, bay.distributions.MultivariateNormalDiag))
    self.assertTrue(y.event_shape == (12,))
    self.assertTrue(z.shape == (1, 8, 12))

    rv = RandomVariable(dim=12, posterior='nbd')
    dist = rv.create_posterior()
    y = dist(x)
    self.assertFalse(rv.is_zero_inflated)
    self.assertTrue(
        isinstance(dist.posterior_layer, bay.layers.NegativeBinomialDispLayer))
    self.assertTrue(dist.prior is None)
    self.assertTrue(
        isinstance(y.distribution, bay.distributions.NegativeBinomialDisp))

    rv = RandomVariable(dim=12, posterior='zinb')
    dist = rv.create_posterior()
    y = dist(x)
    self.assertTrue(rv.is_zero_inflated)
    self.assertTrue(
        isinstance(dist.posterior_layer, bay.layers.ZINegativeBinomialLayer))
    self.assertTrue(dist.prior is None)
    self.assertTrue(isinstance(y.distribution, bay.distributions.ZeroInflated))
    self.assertTrue(
        isinstance(y.distribution.count_distribution,
                   bay.distributions.NegativeBinomial))

    rv = RandomVariable(dim=12, posterior='mse')
    dist = rv.create_posterior()
    y = dist(x)
    self.assertFalse(rv.is_zero_inflated)
    self.assertTrue(rv.is_deterministic)
    self.assertTrue(isinstance(y, bay.distributions.VectorDeterministic))
    z = np.random.rand(*x.shape).astype(x.dtype)
    d1 = -dist.log_prob(z).numpy().ravel()
    d2 = tf.losses.mse(z, y.mean()).numpy().ravel()
    self.assertTrue(np.all(d1 == d2))

  def test_unsupervised_fit_predict(self):
    sco = get_dataset(_DS)
    train, test = sco.split()
    self.assertTrue(sco.n_omics >= 2)
    dca = DeepCountAutoencoder(outputs=RandomVariable(dim=sco.n_vars,
                                                      posterior='mse'),
                               latent_dim=10)
    dca.fit(train, epochs=_EPOCHS, verbose=False)
    dca.fit(train.numpy(), epochs=_EPOCHS, verbose=False)
    self._loss_not_rise(dca.train_history['loss'])
    self._loss_not_rise(dca.valid_history['val_loss'])

    pX, qZ = dca.predict(test, sample_shape=2, verbose=False)
    self.assertTrue(isinstance(pX, bay.distributions.VectorDeterministic))
    self.assertTrue(pX.batch_shape[0] == 2 and pX.batch_shape[1] == test.n_obs)
    self.assertTrue(isinstance(qZ, bay.distributions.VectorDeterministic))

    X = sco.numpy()[:128]
    pX, qZ = dca.predict(X, sample_shape=2, verbose=False)
    self.assertTrue(isinstance(pX, bay.distributions.VectorDeterministic))
    self.assertTrue(pX.batch_shape[0] == 2 and pX.batch_shape[1] == X.shape[0])
    self.assertTrue(isinstance(qZ, bay.distributions.VectorDeterministic))

  def test_variational_model(self):
    sco = get_dataset(_DS)
    n_genes = sco.n_vars
    n_prots = sco.numpy(OMIC.proteomic).shape[1]
    vae = VariationalAutoEncoder(outputs=[
        RandomVariable(dim=n_genes, posterior='zinb', name=OMIC.transcriptomic),
        RandomVariable(dim=n_prots, posterior='nbd', name=OMIC.proteomic)
    ])
    vae.fit(sco, epochs=_EPOCHS, verbose=False)
    self._loss_not_rise(vae.train_history['loss'])
    self._loss_not_rise(vae.valid_history['val_loss'])

    X = sco.numpy()[:128]
    (pX, pY), qZ = vae.predict(X, sample_shape=2, verbose=False)

    self.assertTrue(isinstance(pX.distribution, bay.distributions.ZeroInflated))
    self.assertTrue(
        isinstance(pX.distribution.count_distribution,
                   bay.distributions.NegativeBinomial))
    self.assertTrue(
        isinstance(pY.distribution, bay.distributions.NegativeBinomialDisp))
    self.assertTrue(pX.batch_shape[0] == 2 and pX.batch_shape[1] == X.shape[0])
    self.assertTrue(pY.batch_shape[0] == 2 and pY.batch_shape[1] == X.shape[0])
    self.assertTrue(isinstance(qZ, bay.distributions.MultivariateNormalDiag))
    self.assertTrue(qZ.sample().shape == (X.shape[0],
                                          vae.latents[0].event_shape[0]))

  def test_semi_supervised(self):
    sco = get_dataset(_DS)
    n_genes = sco.n_vars
    n_prots = sco.numpy(OMIC.proteomic).shape[1]
    sisua = SISUA(rna_dim=n_genes, adt_dim=n_prots, alternative_nb=True)
    sisua.fit(sco, epochs=_EPOCHS, verbose=False)
    self._loss_not_rise(sisua.train_history['loss'])
    self._loss_not_rise(sisua.valid_history['val_loss'])

    X = sco.numpy()[:128]
    (pX, pY), qZ = sisua.predict(X, sample_shape=2, verbose=False)

    self.assertTrue(isinstance(pX.distribution, bay.distributions.ZeroInflated))
    self.assertTrue(
        isinstance(pX.distribution.count_distribution,
                   bay.distributions.NegativeBinomialDisp))
    self.assertTrue(
        isinstance(pY.distribution, bay.distributions.NegativeBinomialDisp))
    self.assertTrue(pX.batch_shape[0] == 2 and pX.batch_shape[1] == X.shape[0])
    self.assertTrue(pY.batch_shape[0] == 2 and pY.batch_shape[1] == X.shape[0])
    self.assertTrue(isinstance(qZ, bay.distributions.MultivariateNormalDiag))
    self.assertTrue(
        qZ.sample(1).shape == (1, X.shape[0], sisua.latents[0].event_shape[0]))

  def test_scvi(self):
    sco = get_dataset(_DS)
    train, test = sco.split()
    scvi = SCVI(RandomVariable(sco.n_vars, posterior='zinbd', name='rna'))
    scvi.fit(train, epochs=_EPOCHS, verbose=False)
    pX, (qZ, qL) = scvi.predict(test, verbose=False)

    self._loss_not_rise(scvi.train_history['loss'])
    self._loss_not_rise(scvi.valid_history['val_loss'])

    self.assertTrue(isinstance(pX.distribution, bay.distributions.ZeroInflated))
    self.assertTrue(
        isinstance(pX.distribution.count_distribution,
                   bay.distributions.NegativeBinomialDisp))
    self.assertTrue(pX.batch_shape[0] == 1 and pX.batch_shape[1] == test.n_obs)

    self.assertTrue(isinstance(qZ, bay.distributions.MultivariateNormalDiag))
    self.assertTrue(
        qZ.sample(1).shape == (1, test.n_obs, scvi.latents[0].event_shape[0]))

    self.assertTrue(isinstance(qL.distribution, bay.distributions.Normal))
    self.assertTrue(qL.sample(1).shape == (1, test.n_obs, 1))


if __name__ == '__main__':
  unittest.main()

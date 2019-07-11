from six import add_metaclass
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (Dense, BatchNormalization, Activation,
                                            Dropout, Layer, Concatenate)
from tensorflow_probability.python import distributions as tfd

from odin.backend import is_tensor, get_value
from odin.utils import as_tuple, struct
from odin.backend.keras_helpers import copy_keras_metadata, tied_session
from odin.bay import (MultivariateNormal, ZeroInflatedNegativeBinomial,
                      kl_divergence, ZeroInflated, NegativeBinomial,
                      Bernoulli, Poisson, ZeroInflatedPoisson,
                      update_convert_to_tensor_fn,
                      Moments, ReduceMean, Stddev, GetDistributionAttr, Sampling)

from sisua.models.base import BioModel
from sisua.models.helpers import (get_kl_weight, get_masked_supervision)
# library size
# 'L', 'L_stddev'
# W is predicted corrupted count (with dropout)
# 'W', 'W_stddev'
# V is the cleaned count (without dropout)
# 'V', 'V_stddev'
# 'y_', : predict values of protein markers
# 'pi' : the zero inflated rate
# ===========================================================================
# Helpers
# ===========================================================================
def _layers(hdim, batchnorm=True, n_layers=2,
            input_drop=0, output_drop=0,
            name=''):
  """ Fast utility for creating the feedforward network """
  layers = []
  for i in range(n_layers):
    if 0. < input_drop < 1.:
      layers.append(
          Dropout(rate=input_drop, seed=5218))
    layers.append(
        Dense(int(hdim),
              activation='linear' if batchnorm else 'relu',
              use_bias=False if batchnorm else True))
    if batchnorm:
      layers.append(BatchNormalization())
      layers.append(Activation('relu'))
    if 0. < output_drop < 1.:
      layers.append(
          Dropout(rate=output_drop, seed=5218))
  return Sequential(layers, name=name)

def _output(n_output, dist, name="OutputNet"):
  """ Parse output distribution """
  if dist == 'zinb':
    return Sequential([
        Dense(ZeroInflatedNegativeBinomial.params_size(n_output)),
        ZeroInflatedNegativeBinomial(event_shape=n_output,
                                     given_log_count=True)
    ], name=name)
  elif dist == 'nb':
    return Sequential([
        Dense(NegativeBinomial.params_size(n_output)),
        NegativeBinomial(event_shape=n_output, given_log_count=True)
    ], name=name)
  elif dist == 'poisson':
    return Sequential([
        Dense(Poisson.params_size(n_output)),
        Poisson(event_shape=n_output)
    ], name=name)
  elif dist == 'zipoisson':
    return Sequential([
        Dense(ZeroInflatedPoisson.params_size(n_output)),
        ZeroInflatedPoisson(event_shape=n_output)
    ], name=name)
  elif dist == 'bernoulli':
    return Sequential([
        Dense(Bernoulli.params_size(n_output)),
        Bernoulli(event_shape=n_output)
    ], name=name)
  else:
    raise RuntimeError("No support for distribution: %s" % dist)

def _latent(zdim, n_mcmc_sample, name=None):
  """ Latent posterior

  Returns
  -------
  q_Z_given_X or q_Z_given_Y
  """
  if name is None:
    name = "Latent"
  return Sequential([
      Dense(MultivariateNormal.params_size(event_size=zdim,
                                           covariance_type='diag')),
      MultivariateNormal(event_size=zdim, covariance_type='diag',
            softplus_scale=True,
            convert_to_tensor_fn=lambda d:d.sample(n_mcmc_sample))
  ], name=name)

def _pZ(zdim):
  """ Latent prior """
  return tfd.MultivariateNormalDiag(loc=tf.zeros(shape=zdim),
                                    scale_identity_multiplier=1,
                                    name="p_Z")

def _outputs(q_Z_given_X, p_X_given_Z,
             loss, metrics,
             q_Z_given_Y=None, q_L_given_X=None,
             p_y_given_Z=None):
  """ Extract relevant outputs from each component of the network """
  assert is_tensor(loss), "Loss must be a Tensor"
  out = {
      'loss': copy_keras_metadata(
          p_X_given_Z,
          tf.identity(loss, name='loss')),
      'metr': [copy_keras_metadata(p_X_given_Z, m)
               for m in as_tuple(metrics)]}
  # latent space
  if q_Z_given_Y is None:
    out['Z'] = Moments(variance=False, name="Z")(q_Z_given_X)
  else:
    q_Z_given_XY = Concatenate(axis=1, name='q_Z_given_XY')(
        [Moments(variance=False)(q_Z_given_X),
         Moments(variance=False)(q_Z_given_Y)])
    out['Z'] = q_Z_given_XY
  # this is a tfd.Distribution, a keras.Layer and a tensorflow.Tensor
  # a bit confusing!
  out['Z_sample'] = Sampling(name='Z_sample')(q_Z_given_X)
  # reconstructed values
  out['W_sample'] = Sampling(name='W_sample')(p_X_given_Z)
  out['W'] = ReduceMean(axis=0, name='W')(
      Moments(variance=False)(p_X_given_Z))
  out['W_stddev'] = ReduceMean(axis=0, name='W_stddev')(
      Stddev()(p_X_given_Z))
  # get the denoised
  p_V = p_X_given_Z
  if isinstance(p_V, tfd.Independent):
    p_V = GetDistributionAttr('distribution')(p_X_given_Z)
  if isinstance(p_V, ZeroInflated):
    pi = GetDistributionAttr('inflated_distribution.probs')(p_V)
    out['pi'] = ReduceMean(axis=0, name='pi')(pi)
  if hasattr(p_V, 'count_distribution'):
    p_V = GetDistributionAttr('count_distribution')(p_V)
  # denoised outputs
  out['V_sample'] = Sampling(name='V_sample')(p_V)
  out['V'] = ReduceMean(axis=0, name='V')(
      Moments(variance=False)(p_V))
  out['V_stddev'] = ReduceMean(axis=0, name='V_stddev')(
      Stddev()(p_V))
  # library size
  if q_L_given_X is not None:
    out['L_sample'] = Sampling(name='L_sample')(q_L_given_X)
    out['L'] = Moments(variance=False, name="L")(q_L_given_X)
    out['L_stddev'] = Stddev("L_stddev")(q_L_given_X)
  # semi-supervised
  if p_y_given_Z is not None:
    out['y_sample'] = Sampling(name='y_sample')(p_y_given_Z)
    out['y'] = ReduceMean(axis=0, name="y")(
        Moments(variance=False)(p_y_given_Z))
  return out

# ===========================================================================
# Main model
# ===========================================================================
class vae(BioModel):

  def _init(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    self.encoder_layer = _layers(
        hdim=configs['hdim'], batchnorm=configs['batchnorm'],
        n_layers=configs['nlayer'],
        input_drop=configs['xdrop'], output_drop=configs['edrop'],
        name='Encoder')
    self.latent_layer = _latent(configs['zdim'], nsample)
    self.decoder_layer = _layers(
        hdim=configs['hdim'], batchnorm=configs['batchnorm'],
        n_layers=configs['nlayer'],
        input_drop=configs['zdrop'], output_drop=configs['ddrop'],
        name='Decoder')
    self.output_layer = _output(
        n_output=configs['gene_dim'], dist=configs['xdist'])

  def _call(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    is_training = configs.get('training', True)
    p_Z = _pZ(zdim=configs['zdim'])
    # ====== applying the vae ====== #
    E = self.encoder_layer(X, training=is_training)
    q_Z_given_X = self.latent_layer(E, training=is_training)
    D = self.decoder_layer(q_Z_given_X, training=is_training)
    p_X_given_Z = self.output_layer(D, training=is_training)
    # ====== return ====== #
    KLqp = tf.identity(
        kl_divergence(q_Z_given_X, p_Z,
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp')
    LLK = p_X_given_Z.log_prob(X)
    kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
    ELBO = tf.reduce_mean(
        tf.reduce_logsumexp(
            LLK - kl_weight * KLqp,
            axis=0))
    loss = -ELBO
    return _outputs(q_Z_given_X, p_X_given_Z,
                    loss=loss,
                    metrics=[tf.reduce_mean(LLK, name='LLK'),
                             tf.reduce_mean(KLqp, name='KLqp')])

# ===========================================================================
# Semi-supervised Multi-output
# ===========================================================================
class movae(BioModel):

  def _init(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    self.encoder_layer = _layers(
        hdim=configs['hdim'], batchnorm=configs['batchnorm'],
        n_layers=configs['nlayer'],
        input_drop=configs['xdrop'], output_drop=configs['edrop'],
        name='Encoder')
    self.latent_layer = _latent(configs['zdim'], nsample)
    self.decoder_layer = _layers(
        hdim=configs['hdim'], batchnorm=configs['batchnorm'],
        n_layers=configs['nlayer'],
        input_drop=configs['zdrop'], output_drop=configs['ddrop'],
        name='Decoder')

    self.output_gene_layer = _output(
        n_output=configs['gene_dim'], dist=configs['xdist'],
        name='GeneOutput')
    self.output_prot_layer = _output(
        n_output=configs['prot_dim'], dist=configs['ydist'],
        name='ProteinOutput')

  def _call(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    is_training = configs.get('training', True)
    p_Z = _pZ(zdim=configs['zdim'])
    # ====== applying the vae ====== #
    E = self.encoder_layer(X, training=is_training)
    q_Z_given_X = self.latent_layer(E, training=is_training)
    D = self.decoder_layer(q_Z_given_X, training=is_training)

    p_X_given_Z = self.output_gene_layer(D, training=is_training)
    p_y_given_Z = self.output_prot_layer(D, training=is_training)
    # ====== return ====== #
    KLqp = tf.identity(
        kl_divergence(q_Z_given_X, p_Z,
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp')

    LLK_X = p_X_given_Z.log_prob(X)
    LLK_Y = p_y_given_Z.log_prob(y)

    kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
    ELBO = tf.reduce_mean(
        tf.reduce_logsumexp(
            LLK_X + get_masked_supervision(LLK_Y, mask, nsample, configs['y_weight']) -
            kl_weight * KLqp,
            axis=0))
    loss = -ELBO
    return _outputs(q_Z_given_X, p_X_given_Z,
                    p_y_given_Z=p_y_given_Z,
                    loss=loss,
                    metrics=[tf.reduce_mean(LLK_X, name='LLK_X'),
                             tf.reduce_mean(LLK_Y, name='LLK_Y'),
                             tf.reduce_mean(KLqp, name='KLqp')])

class dovae(BioModel):

  def _init(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    self.encoder_layers = [
        _layers(hdim=configs['hdim'], batchnorm=configs['batchnorm'],
                n_layers=configs['nlayer'],
                input_drop=configs['xdrop'], output_drop=configs['edrop'],
                name=name)
        for name in ('GeneEncoder', 'ProteinEncoder')]

    self.latent_layer = _latent(configs['zdim'], nsample)

    self.decoder_layers = [
        _layers(hdim=configs['hdim'], batchnorm=configs['batchnorm'],
                n_layers=configs['nlayer'],
                input_drop=configs['zdrop'], output_drop=configs['ddrop'],
                name=name)
        for name in ('GeneDecoder', 'ProteinDecoder')]

    self.output_gene_layer = _output(
        n_output=configs['gene_dim'], dist=configs['xdist'],
        name='GeneOutput')
    self.output_prot_layer = _output(
        n_output=configs['prot_dim'], dist=configs['ydist'],
        name='ProteinOutput')

  def _call(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    is_training = configs.get('training', True)
    p_Z = _pZ(zdim=configs['zdim'])
    # ====== applying the vae ====== #
    E_X = self.encoder_layers[0](X, training=is_training)
    E_Y = self.encoder_layers[1](y, training=is_training)

    q_Z_given_X = self.latent_layer(E_X, training=is_training)
    q_Z_given_Y = self.latent_layer(E_Y, training=is_training)

    D_X = self.decoder_layers[0](q_Z_given_X, training=is_training)
    D_Y = self.decoder_layers[1](q_Z_given_Y, training=is_training)

    p_X_given_Z = self.output_gene_layer(D_X, training=is_training)
    p_y_given_Z = self.output_prot_layer(D_Y, training=is_training)
    # ====== return ====== #
    KLqp_X = tf.identity(
        kl_divergence(q_Z_given_X, p_Z,
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp_X')
    KLqp_Y = tf.identity(
        kl_divergence(q_Z_given_Y, p_Z,
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp_Y')

    LLK_X = tf.identity(p_X_given_Z.log_prob(X), name = 'LLK_X')
    LLK_Y = tf.identity(p_y_given_Z.log_prob(y), name = 'LLK_Y')

    kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
    ELBO = tf.reduce_mean(
        tf.reduce_logsumexp(
            LLK_X +
            get_masked_supervision(LLK_Y, mask, nsample, configs['y_weight']) -
            kl_weight * (KLqp_X + KLqp_Y),
            axis=0))
    loss = -ELBO

    return _outputs(q_Z_given_X, p_X_given_Z,
                    p_y_given_Z=p_y_given_Z,
                    loss=loss,
                    metrics=[tf.reduce_mean(LLK_X, name='LLK_X'),
                             tf.reduce_mean(KLqp_X, name='KLqp_X'),
                             tf.reduce_mean(LLK_Y, name='LLK_Y'),
                             tf.reduce_mean(KLqp_Y, name='KLqp_Y')])

# ===========================================================================
# Multi-latent
# ===========================================================================
class mlvae(BioModel):

  def _init(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    self.encoder_layer = _layers(
        hdim=configs['hdim'], batchnorm=configs['batchnorm'],
        n_layers=configs['nlayer'],
        input_drop=configs['xdrop'], output_drop=configs['edrop'],
        name='Encoder')

    self.latent_layers = [
        _latent(configs['zdim'] // 2, nsample, name=name)
        for name in ("GeneLatent", "ProteinLatent")]

    # only separated decoder and output layers
    self.decoder_layers = [
        _layers(hdim=configs['hdim'], batchnorm=configs['batchnorm'],
                n_layers=configs['nlayer'],
                input_drop=configs['zdrop'], output_drop=configs['ddrop'],
                name=name)
        for name in ('GeneDecoder', 'ProteinDecoder')]

    self.output_gene_layer = _output(
        n_output=configs['gene_dim'], dist=configs['xdist'],
        name='GeneOutput')
    self.output_prot_layer = _output(
        n_output=configs['prot_dim'], dist=configs['ydist'],
        name='ProteinOutput')

  def _call(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    is_training = configs.get('training', True)
    # ====== applying the vae ====== #
    E = self.encoder_layer(X, training=is_training)

    q_Z_given_X = self.latent_layers[0](E, training=is_training)
    q_Z_given_Y = self.latent_layers[1](E, training=is_training)

    D_X = self.decoder_layers[0](q_Z_given_X, training=is_training)
    D_Y = self.decoder_layers[1](q_Z_given_Y, training=is_training)

    p_X_given_Z = self.output_gene_layer(D_X, training=is_training)
    p_y_given_Z = self.output_prot_layer(D_Y, training=is_training)
    # ====== return ====== #
    KLqp_X = tf.identity(
        kl_divergence(q_Z_given_X, _pZ(zdim=configs['zdim'] // 2),
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp_X')
    KLqp_Y = tf.identity(
        kl_divergence(q_Z_given_Y, _pZ(zdim=configs['zdim'] // 2),
                      use_analytic_kl=configs['analytic'],
                      q_sample=lambda q: q.sample(nsample),
                      reduce_axis=()),
        name='KLqp_Y')
    kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])

    LLK_X = p_X_given_Z.log_prob(X)
    LLK_Y = p_y_given_Z.log_prob(y)

    ELBO = tf.reduce_mean(
        tf.reduce_logsumexp(
            LLK_X +
            get_masked_supervision(LLK_Y, mask, nsample, configs['y_weight']) -
            kl_weight * (KLqp_X + KLqp_Y),
            axis=0))
    loss = -ELBO
    return _outputs(q_Z_given_X, p_X_given_Z,
                    q_Z_given_Y=q_Z_given_Y,
                    p_y_given_Z=p_y_given_Z,
                    loss=loss,
                    metrics=[tf.reduce_mean(LLK_X, name='LLK_X'),
                             tf.reduce_mean(LLK_Y, name='LLK_Y'),
                             tf.reduce_mean(KLqp_X, name='KLqp_X'),
                             tf.reduce_mean(KLqp_Y, name='KLqp_Y')])

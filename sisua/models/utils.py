from __future__ import absolute_import, division, print_function

import types
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow_probability.python import distributions as tfd

from odin.bay import layers as obl
from odin.bay.distribution_alias import parse_distribution
from odin.bay.layers import (CategoricalLayer, DenseDistribution,
                             MixtureDensityNetwork, OneHotCategoricalLayer,
                             VectorDeterministicLayer)
from odin.networks import ConvNetwork, DeconvNetwork, DenseNetwork, Identity
from sisua.data import SingleCellOMIC


def _recover_mcmc_dim(self, inputs, training=None, mask=None):
  # This is a little hack to ignore MCMC dimension in the decoder
  shape = tf.shape(inputs)
  ndims = len(inputs.shape)
  if ndims > 2:
    inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))
  outputs = super(keras.Sequential, self).call(inputs, training, mask)
  if ndims > 2:
    outputs = tf.reshape(outputs, (shape[0], shape[1], outputs.shape[-1]))
  return outputs


class NetworkConfig(dict):
  r""" A dictionary for storing the networks (encoder and decoder)
  configuration

  Arguments:
    hidden_dim : An Integer, number of hidden units for each hidden layers
    nlayers : An Integer, number of hidden layers
    activation : a String, alias of activation function
    input_dropout : A Scalar [0., 1.], dropout rate, if 0., turn-off dropout.
      this rate is applied for input layer.
     - encoder_dropout : for the encoder output
     - latent_dropout : for the decoder input (right after the latent)
     - decoder_dropout : for the decoder output
     - layer_dropout : for each hidden layer
    batchnorm : A Boolean, batch normalization
    linear_decoder : A Boolean, if `True`, use an `Identity` (i.e. Linear)
      decoder
    pyramid : A Boolean, if `True`, use pyramid structure where the number of
      hidden units decrease as the depth increase
    use_conv : A Boolean, if `True`, use convolutional encoder and decoder
    kernel_size : An Integer, kernel size for convolution network
    strides : An Integer, stride step for convoltion
    conv_proj : An Integer, number of hidden units for the `Dense` projection
      layer right after convolutional network.
  """

  def __init__(self,
               hidden_dim=64,
               nlayers=2,
               activation='relu',
               input_dropout=0.,
               encoder_dropout=0.,
               latent_dropout=0.,
               decoder_dropout=0.,
               layer_dropout=0.,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               kernel_size=5,
               strides=2,
               conv_proj=128):
    kwargs = locals()
    del kwargs['self']
    del kwargs['__class__']
    super().__init__(**kwargs)

  def __getattr__(self, name):
    try:
      attr = super().__getitem__(name)
    except KeyError:
      attr = super().__getattr__(name)
    return attr

  def create_network(self, input_dim, latent_dim, name=None):
    input_dim = int(input_dim)
    latent_dim = int(latent_dim)
    encoder_name = None if name is None else "%s_%s" % (name, "encoder")
    decoder_name = None if name is None else "%s_%s" % (name, "decoder")
    input_shape = (input_dim,)
    latent_shape = (latent_dim,)
    # ====== network config ====== #
    if self.pyramid:
      units = [int(self.hidden_dim / 2**i) for i in range(1, self.nlayers + 1)]
    else:
      units = [self.hidden_dim] * self.nlayers
    # ====== convolution network ====== #
    if self.use_conv:
      encoder = ConvNetwork(units[::-1],
                            rank=1,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding='same',
                            dilation_rate=1,
                            activation=self.activation,
                            use_bias=True,
                            batchnorm=self.batchnorm,
                            input_dropout=self.input_dropout,
                            output_dropout=self.encoder_dropout,
                            layer_dropout=self.layer_dropout,
                            start_layers=[
                                keras.layers.Lambda(
                                    lambda x: tf.expand_dims(x, axis=-1),
                                    output_shape=(input_dim, 1),
                                    input_shape=input_shape,
                                    name='ExpandDims')
                            ],
                            end_layers=[
                                keras.layers.Flatten(),
                                keras.layers.Dense(self.conv_proj,
                                                   activation=self.activation,
                                                   use_bias=True)
                            ],
                            name=encoder_name)
      eshape = encoder.layers[-3].output_shape[1:]
      if not self.linear_decoder:
        decoder = DeconvNetwork(
            units[1:] + [1],
            rank=1,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            dilation_rate=1,
            activation=self.activation,
            use_bias=True,
            batchnorm=self.batchnorm,
            input_dropout=self.latent_dropout,
            output_dropout=self.decoder_dropout,
            layer_dropout=self.layer_dropout,
            start_layers=[
                keras.layers.Dense(self.conv_proj,
                                   activation=self.activation,
                                   use_bias=True,
                                   input_shape=latent_shape),
                keras.layers.Dense(np.prod(eshape),
                                   activation=self.activation,
                                   use_bias=True),
                keras.layers.Reshape(eshape),
            ],
            end_layers=[keras.layers.Flatten()],
            name=decoder_name)
        decoder.call = types.MethodType(_recover_mcmc_dim, decoder)
    # ====== dense network ====== #
    else:
      encoder = DenseNetwork(units=units,
                             activation=self.activation,
                             use_bias=True,
                             batchnorm=self.batchnorm,
                             input_dropout=self.input_dropout,
                             output_dropout=self.encoder_dropout,
                             layer_dropout=self.layer_dropout,
                             input_shape=input_shape,
                             name=encoder_name)
      if not self.linear_decoder:
        decoder = DenseNetwork(units=units[::-1],
                               activation=self.activation,
                               use_bias=True,
                               batchnorm=self.batchnorm,
                               input_dropout=self.latent_dropout,
                               output_dropout=self.decoder_dropout,
                               layer_dropout=self.layer_dropout,
                               input_shape=latent_shape,
                               name=decoder_name)
      decoder.call = types.MethodType(_recover_mcmc_dim, decoder)
    # ====== linear decoder ====== #
    if self.linear_decoder:
      decoder = Identity(name=decoder_name, input_shape=latent_shape)
    # ====== return ====== #
    return encoder, decoder


class RandomVariable:
  r""" Description of a random variable for the model, the variable could
    be the outputs (mRNA, ADT, ...) or the latents.

  Arguments:
    dim : An Integer, number of features (all OMIC data must be 2-D)
    posterior : alias for posterior distribution, or loss function named in
      `tensorflow.losses` or `keras.activations`, for examples:
      - 'bern' : `Bernoulli` distribution
      - ('pois', 'poisson'): `Poisson` distribution
      - ('norm', 'gaus') : `IndependentGaussian` distribution
      - 'diag' : diagonal multivariate Gaussian distribution
      - 'tril' : full (or lower triangle) multivariate Gaussian distribution
      - 'lognorm' : LogNormal distribution
      - 'nb' : negative binomial
      - 'nbd' : negative binomial using mean-dispersion parameterization
      - 'zinb' or 'zinbd' : zero-inflated negative binomial
      - 'mdn' : mixture density network (`IndependentNormal` components)
      - 'mixdiag' : mixture of multivariate diagonal normals
      - 'mixtril' : mixture of multivariate full or triL (lower-triangle) normals
      - 'mse' : deterministic distribution with mean squared error `log_prob`
      - 'vdeterministic' : vectorized deterministic distribution
    name : identity of the OMIC
    kwargs : keyword arguments for initializing the `DistributionLambda`
      of the posterior.
  """

  def __init__(self, dim, posterior='gaus', name=None, prior=None, **kwargs):
    if isinstance(dim, SingleCellOMIC):
      dim = dim.shape[1]
    self.name = name
    self.dim = int(dim)
    self.posterior = str(posterior).lower()
    self.kwargs = kwargs
    self.prior = prior

  def copy(self, posterior=None, **kwargs):
    posterior = self.posterior if posterior is not None else posterior
    kwargs.update(self.kwargs)
    return RandomVariable(self.dim,
                          posterior=posterior,
                          name=self.name,
                          **kwargs)

  def __str__(self):
    return "<RV dim:%d name:%s posterior:%s zi:%s>" % \
      (self.dim, str(self.name), self.posterior, self.is_zero_inflated)

  def __repr__(self):
    return self.__str__()

  @property
  def is_zero_inflated(self):
    return 'zi' == self.posterior[:2].lower()

  @property
  def is_deterministic(self):
    if self.posterior in dir(tf.losses) or \
      self.posterior in dir(keras.activations):
      return True
    return False

  @property
  def input_shape(self):
    return (self.dim,)

  def create_posterior(self):
    r""" Initiate a Distribution for the random variable """
    prior = self.prior
    # ====== deterministic distribution with loss function from tensorflow ====== #
    if self.posterior in dir(tf.losses) or \
      self.posterior in dir(keras.activations):
      distribution = VectorDeterministicLayer
      if self.posterior in dir(tf.losses):
        activation = 'relu'
        fn = tf.losses.get(str(self.posterior))
      else:
        activation = keras.activations.get(self.posterior)
        fn = tf.losses.get('mse')
      llk_fn = lambda self, y_true: tf.expand_dims(
          -fn(y_true, self.posterior.mean()), axis=-1)
    # ====== probabilistic loss ====== #
    else:
      if self.posterior in ('mdn', 'mixdiag', 'mixfull'):
        distribution = None
      else:
        distribution = parse_distribution(self.posterior)[0]
      activation = 'linear'
      llk_fn = lambda self, y_true: tf.expand_dims(
          self.posterior.log_prob(y_true), axis=-1)
      # special case for the prior
      if prior is None and distribution is not None:
        if distribution == obl.GaussianLayer:
          prior = tfd.Independent(
              tfd.Normal(loc=tf.zeros(shape=[self.dim]),
                         scale=tf.ones(shape=[self.dim])), 1)
        elif issubclass(distribution, obl.MultivariateNormalLayer):
          cov = distribution._partial_kwargs['covariance']
          if cov == 'diag':
            prior = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=[self.dim]),
                                               scale_identity_multiplier=1)
          else:
            bijector = tfp.bijectors.FillScaleTriL(
                diag_bijector=tfp.bijectors.Identity(), diag_shift=1e-5)
            scale_tril = bijector.forward(
                tf.ones([self.dim * (self.dim + 1) // 2]))
            prior = tfd.MultivariateNormalTriL(loc=tf.zeros(shape=[self.dim]),
                                               scale_tril=scale_tril)
        elif distribution == obl.LogNormalLayer:
          prior = tfd.Independent(
              tfd.LogNormal(loc=tf.zeros(shape=[self.dim]),
                            scale=tf.ones(shape=[self.dim])), 1)
    # ====== create distribution layers ====== #
    kwargs = dict(self.kwargs)
    activation = kwargs.pop('activation', activation)
    if self.posterior in ('mdn', 'mixdiag', 'mixfull', 'mixtril'):
      kwargs.pop('covariance', None)
      layer = MixtureDensityNetwork(self.dim,
                                    loc_activation=activation,
                                    scale_activation='softplus1',
                                    covariance=dict(
                                        mdn='none',
                                        mixdiag='diag',
                                        mixfull='tril',
                                        mixtril='tril')[self.posterior],
                                    name=self.name,
                                    **kwargs)
      layer.set_prior()
    else:
      layer = DenseDistribution(self.dim,
                                posterior=distribution,
                                prior=prior,
                                activation=activation,
                                posterior_kwargs=kwargs,
                                name=self.name)
    layer.log_prob = types.MethodType(llk_fn, layer)
    return layer

from __future__ import print_function, division, absolute_import

import re

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb
from tensorflow_probability import bijectors as tfb

from odin.maths import edit_distance
from odin import nnet as N, backend as K
from odin.utils import ctext

from sisua.distributions.zero_inflated import ZeroInflated
from sisua import is_verbose

_space_char = re.compile(r"\s")
_non_alphanumeric_char = re.compile(r"\W")

def _normalize_text(text):
  text = str(text).strip().lower()
  text = re.sub(r"\s\s+", " ", text)
  text = _space_char.sub(' ', text)
  text = _non_alphanumeric_char.sub(' ', text)
  return text

# ===========================================================================
# Others
# ===========================================================================
def get_kl_weight(nepoch, weight, warmup):
  """
  nepoch   : use for calculating warm-up weight
  weight   : a fixed scalar
  warmup  : number of warm-up epoch
  """
  with tf.name_scope("kl_weight"):
    warmup = tf.convert_to_tensor(warmup, 'float32')
    weight = tf.convert_to_tensor(weight, 'float32')
    nepoch = tf.convert_to_tensor(nepoch, 'float32')
    warm_up_weight = tf.minimum(tf.maximum(nepoch, 1.) / warmup, 1.)
    return warm_up_weight * weight

def get_masked_supervision(y_loss, mask, nsample, y_weight):
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  y_loss = tf.multiply(y_loss, mask)
  return y_weight * y_loss

def extract_pi(outputs, dist):
  if 'pi' in dist.distribution.parameters:
    pi = tf.reduce_mean(dist.distribution.pi, axis=0, name="ZI_PI")
    outputs['pi'] = pi

def extract_clean_count(outputs, dist):
  if isinstance(dist, tfd.Independent):
    dist = dist.distribution
  if isinstance(dist, ZeroInflated):
    dist = dist.dist
    mean = dist.mean()
    V_expected = tf.identity(tf.reduce_mean(dist.mean(), axis=0),
                             name="V_expected")
    V_stddev_total = tf.identity(
        tf.sqrt(
            tf.reduce_mean(dist.variance(), axis=0)
      ),
    name="V_stddev_total")
    V_stddev_explained = tf.identity(
        tf.sqrt(
            tf.reduce_mean(tf.square(mean - tf.expand_dims(V_expected, axis=0)),
                         axis=0)),
    name="V_stddev_explained")
    outputs['V'] = V_expected
    outputs['V_stddev'] = V_stddev_total

# ===========================================================================
# Network
# ===========================================================================
def create_network(hdim, batchnorm=True, n_layers=2,
                   xdrop=0.3, zdrop=0.2, edrop=0, ddrop=0,
                   name=''):
  name = str(name)
  if len(name) > 0:
    name = '_%s' % name

  def dense_batchnorm(layer_name):
    return N.Sequence([
        N.Dense(num_units=hdim,
                b_init=None if batchnorm else 0,
                activation=K.linear if batchnorm else tf.nn.relu),
        N.BatchNorm(axes='auto', activation=K.relu) if batchnorm else None
    ], debug=False, name=layer_name)

  f_encoder = N.Sequence([
      N.Dropout(level=xdrop) if xdrop > 0 else None,
      [dense_batchnorm("Layer%d" % i) for i in range(n_layers)],
      N.Dropout(level=edrop) if edrop > 0 else None,
  ], debug=is_verbose(), name='EncoderNetwork%s' % name)

  f_decoder = N.Sequence([
      N.Dropout(level=zdrop) if zdrop > 0 else None,
      [dense_batchnorm("Layer%d" % i) for i in range(n_layers)],
      N.Dropout(level=ddrop) if ddrop > 0 else None,
  ], debug=is_verbose(), name="DecoderNetwork%s" % name)

  return f_encoder, f_decoder

# ===========================================================================
# Loss function
# ===========================================================================
def parse_loss_function(true, logit, mask, loss_name):
  """ Return: prediction, loss """
  num_classes = true.shape.as_list()[-1]
  if mask is None:
    mask = 1.0
    mask_2D = 1.0
  else:
    mask_2D = tf.expand_dims(mask, axis=-1)
  # ====== log ====== #
  if is_verbose():
    print(ctext("Parsing loss function:", 'lightyellow'))
    print("  true :", ctext(true, 'cyan'))
    print("  logit:", ctext(logit, 'cyan'))
    print("  mask :", ctext(mask, 'cyan'))
    print("  name :", ctext(loss_name, 'cyan'))
  # ====== search for appropritate loss function ====== #
  if loss_name == 'sg':
    loss = 0
    for i in range(num_classes):
      loss += tf.losses.sigmoid_cross_entropy(multi_class_labels=true[:, i],
                                              logits=logit[:, i],
                                              weights=mask)
    loss = loss / num_classes
    pred = tf.nn.sigmoid(logit)
  elif loss_name == 'ce':
    loss = tf.losses.softmax_cross_entropy(onehot_labels=true,
                                           logits=logit,
                                           weights=mask)
    pred = tf.nn.softmax(logit)
  elif loss_name == 'll':
    pred = tf.nn.sigmoid(logit)
    loss = tf.losses.log_loss(labels=true,
                              predictions=pred,
                              weights=mask_2D)
  elif loss_name == 'mse':
    pred = tf.nn.relu(logit)
    loss = tf.losses.mean_squared_error(labels=true,
                                        predictions=pred,
                                        weights=mask_2D)
  else:
    raise RuntimeError("Unknown loss type: %s" % loss_name)
  if is_verbose():
    print("  loss :", ctext(loss, 'lightcyan'))
    print("  pred :", ctext(pred, 'lightcyan'))
  return pred, loss

# ===========================================================================
# Distribution
# ===========================================================================
_dist_name = [
    'bernoulli',
    'zibernoulli',

    'normal',

    'nb',
    'zinb',

    'poisson',
    'zipoisson',

    'beta',

    'lognb',
]

_epsilon = 1e-6

def clip_support(x, x_min, x_max):
  return tf.clip_by_value(x, x_min + _epsilon, x_max - _epsilon)

def parse_output_distribution(X, D, dist_name, name):
  """ Return: out, expected, stddev_explained, stddev_total, NLLK

  D : tensor
    output from decoder (not included the output layer)
  dist_name : str
    name of the distribution
  out_dim : int
    number of output dimension

  Return
  ------
  """
  dist_name = _normalize_text(dist_name)
  out_dim = X.shape.as_list()[-1]
  d_dim = D.shape.as_list()[-1] # decoder output dimension

  assert dist_name in _dist_name, \
  "Support distribution: %s; given: '%s'" % (', '.join(_dist_name), dist_name)
  # ====== some log ====== #
  if is_verbose():
    print(ctext("Parsing variable distribution:", 'lightyellow'))
    print("  Variable    :", ctext(X, 'cyan'))
    print("  Decoder     :", ctext(D, 'cyan'))
    print("  name        :", ctext('%s/%s' % (dist_name, name), 'cyan'))
  # ******************** create distribution ******************** #
  with tf.variable_scope(name):
    # ====== Bernoulli ====== #
    if dist_name == 'zibernoulli':
      f = N.Dense(num_units=out_dim, activation=K.linear,
                  name="Logit" + name)
      bern = tfd.Bernoulli(logits=f(D))

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid,
                     name="Pi" + name)
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=bern, pi=pi)

    # ====== Bernoulli ====== #
    elif dist_name == 'bernoulli':
      f = N.Dense(num_units=out_dim, activation=K.linear,
                  name="Logit" + name)
      out = tfd.Bernoulli(logits=f(D))

    # ====== Normal ====== #
    elif dist_name == 'normal':
      f_loc = N.Dense(num_units=out_dim, activation=K.linear,
                      name="Location" + name)
      loc = f_loc(D)

      f_log_sigma = N.Dense(num_units=out_dim, activation=K.linear,
                            name="LogSigma" + name)
      log_sigma = clip_support(f_log_sigma(D), x_min=-3, x_max=3)

      out = tfd.Normal(loc=loc, scale=tf.exp(log_sigma))

    # ====== Poisson ====== #
    elif dist_name in ('poisson',):
      f_log_rate = N.Dense(num_units=out_dim, activation=K.linear,
                           name="LogRate" + name)
      out = tfd.Poisson(log_rate=f_log_rate(D), name=name)

    # ====== Zero-inflated Poisson ====== #
    elif dist_name in ('zipoisson',):
      f_log_rate = N.Dense(num_units=out_dim, activation=K.linear,
                           name="LogRate" + name)
      pois = tfd.Poisson(log_rate=f_log_rate(D))

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid,
                     name="Pi" + name)
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=pois, pi=pi)
    # ====== Negative Binomial ====== #
    elif dist_name in ('nb',):
      f_log_count = N.Dense(num_units=out_dim, activation=K.linear,
                            name="TotalCount" + name)
      log_count = clip_support(f_log_count(D), x_min=-10, x_max=10)

      f_logits = N.Dense(num_units=out_dim, activation=K.linear,
                         name="Logits" + name)

      out = tfd.NegativeBinomial(
          total_count=tf.exp(log_count),
          logits=f_logits(D))
    # ====== Zero-inflated Negative Binomial ====== #
    elif dist_name in ('zinb',):
      f_log_count = N.Dense(num_units=out_dim, activation=K.linear,
                            name="TotalCount" + name)
      log_count = clip_support(f_log_count(D), x_min=-10, x_max=10)

      f_logits = N.Dense(num_units=out_dim, activation=K.linear,
                         name="Logits" + name)

      nb = tfd.NegativeBinomial(
          total_count=tf.exp(log_count),
          logits=f_logits(D))

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid,
                     name="Pi" + name)
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=nb, pi=pi)
    # ====== beta distribution ====== #
    elif dist_name in ('beta',):
      f_log_alpha = N.Dense(num_units=out_dim, activation=tf.identity,
                            name="LogAlpha" + name)
      log_alpha = clip_support(f_log_alpha(D), x_min=-3, x_max=3)

      f_log_beta = N.Dense(num_units=out_dim, activation=tf.identity,
                           name="LogBeta" + name)
      log_beta = clip_support(f_log_beta(D), x_min=-3, x_max=3)

      out = tfd.Beta(concentration1=tf.exp(log_alpha),
                     concentration0=tf.exp(log_beta))
    # ====== LogNormal poisson ====== #
    elif dist_name in ('lognb',):
      raise NotImplementedError
    # ====== exception ====== #
    else:
      raise RuntimeError("Cannot find distribution with name: '%s', all possible "
        "distributions are: %s" % (dist_name, str(_dist_name)))
    # ====== independent ====== #
    out = tfd.Independent(out,
                          reinterpreted_batch_ndims=1)
  # ====== get the Negative log-likelihood ====== #
  X_tile = tf.tile(tf.expand_dims(X, axis=0),
                   multiples=[tf.shape(D)[0], 1, 1])
  # negative log likelihood. (n_samples, n_batch)
  # if ndim == 3, sum the NLLK among all features
  NLLK = -out.log_prob(X_tile)
  if NLLK.shape.ndims == 3:
    NLLK = tf.reduce_sum(NLLK, axis=-1)
  # ******************** get the expectation, and stddev ******************** #
  # [n_sample, n_batch, feat_dim]
  mean = out.mean()
  # [n_batch, feat_dim]
  expected = tf.identity(
      tf.reduce_mean(mean, axis=0),
  name="%s_expected" % name)
  # MCMC variance [n_batch, feat_dim] (/ tf.cast(nsample, 'float32') ?)
  stddev_explained = tf.identity(
      tf.sqrt(
          tf.reduce_mean(tf.square(mean - tf.expand_dims(expected, axis=0)),
                         axis=0)),
  name="%s_stddev_explained" % name)
  # analytical variance
  stddev_total = tf.identity(
      tf.sqrt(
          tf.reduce_mean(out.variance(), axis=0)
    ),
  name="%s_stddev_total" % name)
  # ******************** print the dist ******************** #
  if is_verbose():
    print("  dist        :", ctext(out, 'cyan'))
    for name, p in sorted(out.parameters.items()):
      if name in ('allow_nan_stats', 'validate_args'):
        continue
      print("      %-8s:" % name, ctext(p, 'magenta'))
    print("  NLLK        :", ctext(NLLK, 'cyan'))
    print("  Expected    :", ctext(expected, 'cyan'))
    print("  StdExplained:", ctext(stddev_explained, 'cyan'))
    print("  StdTotal    :", ctext(stddev_total, 'cyan'))
  return out, expected, stddev_explained, stddev_total, NLLK

# ===========================================================================
# Latent distribution
# ===========================================================================
_latent_dist = [
    'mixture',
    'diag',
    'normal'
]
def parse_latent_distribution(E, zdim, dist_name, name,
                              n_samples, analytic):
  """
  Parameters
  ----------
  E : [n_batch, n_output_encoder]
    can be a Tensor or list of Tensor of encoder output

  Return
  ------
  list of [
    (
      Z1 : concatenated logits value of distribution parameters
        (n_batch, zdim * n_distribution_parameters)
      qZ_given_X1 : (n_batch, zdim)
      Z_given_X1_samples : (n_samples, n_batch, zdim)
      KL_X1_divergent : ()
    ),
    (
      Z2, qZ_given_X2, Z_given_X2_samples, KL_X2_divergent
    ),
    ...
  ]
  """
  dist_name = _normalize_text(dist_name)
  if dist_name not in _latent_dist:
    raise ValueError("Only support following latent distribution: %s" %
      ', '.join(_latent_dist))
  if not isinstance(E, (tuple, list)):
    E = (E,)
  E = tuple(E)
  n_inputs = len(E)
  # ====== some log ====== #
  if is_verbose():
    print(ctext("Parsing LATENT distribution:", 'lightyellow'))
    print("  name        :", ctext('%s/%s' % (dist_name, name), 'cyan'))
    print("  Encoder     :", ctext(E, 'cyan'))
    print("  zdim        :", ctext(zdim, 'cyan'))
    print("  n_samples   :", ctext(n_samples, 'cyan'))
    print("  analytic    :", ctext(analytic, 'cyan'))
  # ====== store multiple results ====== #
  Z_ = []
  qZ_given_ = []
  KL_ = []
  # ******************** create the posterior and prior ******************** #
  with tf.variable_scope(name):
    # ====== Normal ====== #
    if dist_name == 'normal':
      # prior
      pZ = tfd.Normal(loc=tf.zeros(shape=(1,)), scale=tf.ones(shape=(1,)),
                      name='pZ')
      # posterior
      f_loc = N.Dense(num_units=zdim, activation=K.linear,
                      name='Loc' + name)
      f_log_scale = N.Dense(num_units=zdim, activation=K.linear,
                            name='Scale' + name)
      for i, e in enumerate(E):
        loc = f_loc(e)
        log_scale = clip_support(f_log_scale(e), x_min=-3, x_max=3)
        Z_.append(tf.concat([loc, log_scale], axis=-1))
        qZ_given_.append(tfd.Normal(loc=loc, scale=tf.exp(log_scale),
                                    name="qZ_given_X%d" % i))
    # ====== Diag ====== #
    elif dist_name == 'diag':
      # prior
      pZ = tfd.MultivariateNormalDiag(
          loc =tf.zeros(shape=(zdim,)), scale_identity_multiplier=1.0,
          name='pZ')
      # posterior
      f_loc = N.Dense(num_units=zdim, activation=K.linear,
                      name='Loc' + name)
      f_scale = N.Dense(num_units=zdim, activation=K.linear,
                        name='Scale' + name)
      for i, e in enumerate(E):
        loc = f_loc(e)
        scale = f_scale(e)
        Z_.append(tf.concat([loc, scale], axis=-1))
        qZ_given_.append(tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=tf.nn.softplus(scale + K.softplus_inverse(1.0)),
            name="qZ_given_X%d" % i))
    # ====== Mixture of Gaussian ====== #
    elif dist_name == 'mixture':
      raise NotImplementedError
  # ====== KL ====== #
  Z_samples_given_ = [q.sample(n_samples) for q in qZ_given_]
  if analytic:
    # [n_batch] or [n_batch, zdim]
    for q in qZ_given_:
      kl = tfd.kl_divergence(q, pZ)
      if kl.shape.ndims > 1:
        kl = tf.reduce_sum(kl, axis=-1, keepdims=False)
      kl = tf.expand_dims(kl, axis=0)
      KL_.append(kl)
  else:
    # [n_samples, n_batch] or [n_samples, n_batch, zdim]
    for q, zsamples in zip(qZ_given_, Z_samples_given_):
      kl = (q.log_prob(zsamples) - pZ.log_prob(zsamples))
      # if 3-D, return KL-divergence among all features dimension
      if kl.shape.ndims == 3:
        kl = tf.reduce_sum(kl, axis=-1, keepdims=False)
      elif kl.shape.ndims > 3:
        raise RuntimeError("KL > 3-D, not possible!")
      KL_.append(kl)
  # ******************** print the dist ******************** #
  if is_verbose():
    for z, qz, zsamples, kl in zip(Z_, qZ_given_, Z_samples_given_, KL_):
      print("  ---")
      print("  Z           :", ctext(z, 'cyan'))
      print("  q(Z|X)      :", ctext(qz, 'cyan'))
      for name, p in sorted(qz.parameters.items()):
        if name in ('allow_nan_stats', 'validate_args'):
          continue
        print("      %-8s:" % name, ctext(p, 'magenta'))
      print("  KL          :", ctext(kl, 'cyan'))
      print("  Samples     :", ctext(zsamples, 'cyan'))

    print("  ---")
    print("  p(Z)        :", ctext(pZ, 'cyan'))
    for name, p in sorted(pZ.parameters.items()):
      if name in ('allow_nan_stats', 'validate_args'):
        continue
      print("      %-8s:" % name, ctext(p, 'magenta'))
  # ====== return ====== #
  ret = tuple(zip(Z_, qZ_given_, Z_samples_given_, KL_))
  return ret[0] if len(ret) == 1 else ret

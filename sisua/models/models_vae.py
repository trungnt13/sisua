from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

from odin import nnet as N, backend as K

from sisua.models.helpers import (parse_latent_distribution,
                                  parse_variable_distribution,
                                  create_network, get_count_sum,
                                  extract_pi, extract_clean_count)

# ===========================================================================
# Main models
# ===========================================================================
@N.Lambda
def vae(X, T, C, mask, y, nsample, kwargs):
  """ M1 model:
  Generative: z -> x
  Inference : X -> z
  """
  # ====== create network ====== #
  f_encoder, f_decoder = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop= kwargs['ddrop'],
      name="VAE")
  # ====== posterior ====== #
  E = f_encoder(X)
  Z, qZ_given_X, Z_given_X_samples, KL = parse_latent_distribution(E,
      zdim=kwargs['zdim'], dist_name=kwargs['zdist'], name='X_latent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])
  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)
  # ====== reconstruction and distortion ====== #
  D = f_decoder(tf.concat([Z_given_X_samples, C], axis=-1, name='Z_N')
                if kwargs['count_sum'] else
                Z_given_X_samples)
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK = \
  parse_variable_distribution(
      X=T, D=D,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL)
  outputs = {'loss': tf.identity(loss, "VAE_loss"),
             'metr': [tf.reduce_mean(NLLK, name="NLLK"),
                      tf.reduce_mean(KL, name="KLqp")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

# ===========================================================================
# Semi-supervised VAE
# ===========================================================================
@N.Lambda
def movae(X, T, C, mask, y, nsample, kwargs):
  # ====== create network ====== #
  f_encoder, f_decoder = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'], zdrop=kwargs['zdrop'],
      name="Multi_output_VAE")
  # ====== posterior ====== #
  E = f_encoder(X)
  Z, qZ_given_X, Z_given_X_samples, KL = parse_latent_distribution(E,
      zdim=kwargs['zdim'], dist_name=kwargs['zdist'], name='X_latent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])
  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)
  # ====== reconstruction and distortion ====== #
  D = f_decoder(tf.concat([Z_given_X_samples, C], axis=-1, name='Z_N')
                if kwargs['count_sum'] else
                Z_given_X_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK_X = \
  parse_variable_distribution(
      X=T, D=D,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stdev_explained, T_stdev_total, NLLK_Y = \
  parse_variable_distribution(
      X=y, D=D,
      dist_name=kwargs['ydist'], name='pY_given_Z')
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  # final negative log likelihood
  NLLK = NLLK_X + tf.multiply(NLLK_Y, mask)
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL)
  outputs = {'loss': tf.identity(loss, "Multi_output_VAE_loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL, name="KLqp")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

@N.Lambda
def dovae(X, T, C, mask, y, nsample, kwargs):
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_output_VAE_X")
  f_encoder_Y, f_decoder_Y = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_output_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)
  E_Y = f_encoder_Y(y)

  (Z_X, qZ_given_X, Z_given_X_samples, KL_X),\
  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=[E_X, E_Y],
      zdim=kwargs['zdim'], dist_name=kwargs['zdist'], name='XY_latent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])
  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)
  # ====== reconstruction and distortion ====== #
  D_X = f_decoder_X(tf.concat([Z_given_X_samples, C], axis=-1, name='ZX_N')
                    if kwargs['count_sum'] else
                    Z_given_X_samples)
  D_Y = f_decoder_Y(tf.concat([Z_given_Y_samples, C], axis=-1, name='ZY_N')
                    if kwargs['count_sum'] else
                    Z_given_Y_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK_X = \
  parse_variable_distribution(
      X=T, D=D_X,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stdev_explained, T_stdev_total, NLLK_Y = \
  parse_variable_distribution(
      X=y, D=D_Y,
      dist_name=kwargs['ydist'], name='pY_given_Z')
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  # final negative log likelihood
  NLLK = NLLK_X + tf.multiply(NLLK_Y, mask)
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL_X + KL_Y, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL_X + KL_Y)
  outputs = {'loss': tf.identity(loss, "Dual_output_VAE_loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

# ===========================================================================
# Semi-supervised latent VAE
# ===========================================================================
@N.Lambda
def mlvae(X, T, C, mask, y, nsample, kwargs):
  """ Multi-latent VAE """
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Multi_latent_VAE_X")
  # we only take the decoder here
  _, f_decoder_Y = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Multi_latent_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)

  (Z_X, qZ_given_X, Z_given_X_samples, KL_X) = \
  parse_latent_distribution(
      E=E_X,
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z1_Xlatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=E_X,
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z2_Ylatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)

  # ====== reconstruction and distortion ====== #
  Z_given_XY_samples = tf.concat((Z_given_X_samples, Z_given_Y_samples),
                                 axis=-1)
  D_X = f_decoder_X(tf.concat([Z_given_XY_samples, C], axis=-1, name='ZX_N')
                    if kwargs['count_sum'] else
                    Z_given_XY_samples)
  D_Y = f_decoder_Y(Z_given_Y_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK_X = \
  parse_variable_distribution(
      X=T, D=D_X,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stdev_explained, T_stdev_total, NLLK_Y = \
  parse_variable_distribution(
      X=y, D=D_Y,
      dist_name=kwargs['ydist'], name='pY_given_Z')
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  # final negative log likelihood
  NLLK = NLLK_X + tf.multiply(NLLK_Y, mask)
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL_X + KL_Y, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL_X + KL_Y)
  outputs = {'loss': tf.identity(loss, "Multi_latent_VAE_loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = tf.concat((qZ_given_X.mean(), qZ_given_Y.mean()),
                           axis=-1)
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

@N.Lambda
def dlvae(X, T, C, mask, y, nsample, kwargs):
  """ Parallel latents VAE """
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_latent_VAE_X")
  f_encoder_Y, f_decoder_Y = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_latent_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)
  E_Y = f_encoder_Y(y)

  (Z_X1, qZ_given_X1, Z_given_X1_samples, KL_X1) = \
  parse_latent_distribution(
      E=E_X,
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z1_Xlatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  (Z_X2, qZ_given_X2, Z_given_X2_samples, KL_X2),\
  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=[E_X, E_Y],
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z2_Ylatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)

  # ====== reconstruction and distortion ====== #
  Z_given_XY_samples = tf.concat((Z_given_X1_samples, Z_given_X2_samples),
                                 axis=-1)
  D_X = f_decoder_X(tf.concat([Z_given_XY_samples, C], axis=-1, name='ZX_N')
                    if kwargs['count_sum'] else
                    Z_given_XY_samples)
  D_Y = f_decoder_Y(Z_given_Y_samples)

  # for X -> W
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK_X = \
  parse_variable_distribution(
      X=T, D=D_X,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stdev_explained, T_stdev_total, NLLK_Y = \
  parse_variable_distribution(
      X=y, D=D_Y,
      dist_name=kwargs['ydist'], name='pY_given_Z')
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  # final negative log likelihood
  NLLK = NLLK_X + tf.multiply(NLLK_Y, mask)
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL_X1 + KL_X2 + KL_Y, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL_X1 + KL_X2 + KL_Y)
  outputs = {'loss': tf.identity(loss, "Dual_latent_VAE_loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X1 + KL_X2, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = tf.concat((qZ_given_X1.mean(), qZ_given_X2.mean()),
                           axis=-1)
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

# ===========================================================================
# Supervised model
# ===========================================================================
@N.Lambda
def ssvae(X, T, C, mask, y, nsample, kwargs):
  """ Parallel latents VAE """
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_latent_VAE_X")
  f_encoder_Y, f_decoder_Y = create_network(
      hdim=kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], edrop=kwargs['edrop'],
      zdrop=kwargs['zdrop'], ddrop=kwargs['ddrop'],
      name="Dual_latent_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)
  E_Y = f_encoder_Y(y)

  (Z_X, qZ_given_X, Z_given_X_samples, KL_X) = \
  parse_latent_distribution(
      E=E_X,
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z1_Xlatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=E_Y,
      zdim=kwargs['zdim'] // 2, dist_name=kwargs['zdist'], name='Z2_Ylatent',
      n_samples=nsample,
      analytic=kwargs['analytic'], constraint=kwargs['constraint'])

  if kwargs['count_sum']:
    C = get_count_sum(X, nsample, kwargs)

  # ====== reconstruction and distortion ====== #
  Z_given_XY_samples = tf.concat((Z_given_X_samples, Z_given_Y_samples),
                                 axis=-1)
  D_X = f_decoder_X(tf.concat([Z_given_XY_samples, C], axis=-1, name='ZX_N')
                    if kwargs['count_sum'] else
                    Z_given_XY_samples)
  D_Y = f_decoder_Y(tf.concat([Z_given_Y_samples, C], axis=-1, name='ZY_N')
                    if kwargs['count_sum'] else
                    Z_given_Y_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stdev_explained, W_stdev_total, NLLK_X = \
  parse_variable_distribution(
      X=T, D=D_X,
      dist_name=kwargs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stdev_explained, T_stdev_total, NLLK_Y = \
  parse_variable_distribution(
      X=y, D=D_Y,
      dist_name=kwargs['ydist'], name='pY_given_Z')
  # apply masking for supervised loss
  mask = tf.tile(tf.expand_dims(mask, axis=0),
                 multiples=[nsample, 1])
  # final negative log likelihood
  NLLK = NLLK_X + tf.multiply(NLLK_Y, mask)
  # ====== elbo ====== #
  if kwargs['iw']:
    loss = tf.reduce_mean(
        tf.reduce_logsumexp(NLLK + KL_X + KL_Y, axis=0) -
        tf.log(tf.cast(nsample, 'float32')))
  else:
    loss = tf.reduce_mean(NLLK + KL_X + KL_Y)
  outputs = {'loss': tf.identity(loss, "Dual_latent_VAE_loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = tf.concat((qZ_given_X.mean(), qZ_given_Y.mean()),
                           axis=-1)
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stdev_total'] = W_stdev_total
  outputs['W_stdev_explained'] = W_stdev_explained
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

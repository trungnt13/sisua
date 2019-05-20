from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

from odin import nnet as N, backend as K

from sisua.models.helpers import (parse_latent_distribution,
                                  parse_output_distribution,
                                  create_network, extract_pi, extract_clean_count,
                                  get_kl_weight, get_masked_supervision)

# ===========================================================================
# Main models
# ===========================================================================
@N.Lambda
def oldvae(X, T, L, L_mean, L_var, mask, y, nsample, nepoch, configs):
  """ M1 model:
  Generative: z -> x
  Inference : X -> z
  """
  # ====== create network ====== #
  f_encoder, f_decoder = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop= configs['ddrop'],
      name="VAE")
  # ====== posterior ====== #
  E = f_encoder(X)
  Z, qZ_given_X, Z_given_X_samples, KL = parse_latent_distribution(E,
      zdim=configs['zdim'], dist_name=configs['zdist'], name='X_latent',
      n_samples=nsample,
      analytic=configs['analytic'])
  # ====== reconstruction and distortion ====== #
  D = f_decoder(Z_given_X_samples)
  pX_given_Z, W_expected, W_stddev_explained, W_stddev_total, NLLK = \
  parse_output_distribution(
      X=T, D=D,
      dist_name=configs['xdist'], name='pX_given_Z')
  # ====== elbo ====== #
  kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
  # logsumexp could return very different loss
  loss = tf.reduce_mean(
      tf.reduce_logsumexp(NLLK + kl_weight * KL, axis=0))
  outputs = {'loss': loss,
             'metr': [tf.reduce_mean(NLLK, name="NLLK"),
                      tf.reduce_mean(KL, name="KLqp")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stddev'] = W_stddev_total
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

# ===========================================================================
# Semi-supervised VAE
# ===========================================================================
@N.Lambda
def oldmovae(X, T, L, L_mean, L_var, mask, y, nsample, nepoch, configs):
  # ====== create network ====== #
  f_encoder, f_decoder = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      xdrop=configs['xdrop'], edrop=configs['edrop'], zdrop=configs['zdrop'],
      name="Multi_output_VAE")
  # ====== posterior ====== #
  E = f_encoder(X)
  Z, qZ_given_X, Z_given_X_samples, KL = parse_latent_distribution(E,
      zdim=configs['zdim'], dist_name=configs['zdist'], name='X_latent',
      n_samples=nsample,
      analytic=configs['analytic'])
  # ====== reconstruction and distortion ====== #
  D = f_decoder(Z_given_X_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stddev_explained, W_stddev_total, NLLK_X = \
  parse_output_distribution(
      X=T, D=D,
      dist_name=configs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stddev_explained, T_stddev_total, NLLK_Y = \
  parse_output_distribution(
      X=y, D=D,
      dist_name=configs['ydist'], name='pY_given_Z')
  # final negative log likelihood
  NLLK = NLLK_X + get_masked_supervision(NLLK_Y, mask, nsample, configs['y_weight'])
  # ====== elbo ====== #
  kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
  loss = tf.reduce_mean(
      tf.reduce_logsumexp(NLLK + kl_weight * KL, axis=0))
  outputs = {'loss': loss,
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL, name="KLqp")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stddev'] = W_stddev_total
  outputs['y'] = tf.reduce_mean(pY_given_Z.mean(), axis=0)
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

@N.Lambda
def olddovae(X, T, L, L_mean, L_var, mask, y, nsample, nepoch, configs):
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Dual_output_VAE_X")
  f_encoder_Y, f_decoder_Y = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Dual_output_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)
  E_Y = f_encoder_Y(y)

  (Z_X, qZ_given_X, Z_given_X_samples, KL_X),\
  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=[E_X, E_Y],
      zdim=configs['zdim'], dist_name=configs['zdist'], name='XY_latent',
      n_samples=nsample,
      analytic=configs['analytic'])
  # ====== reconstruction and distortion ====== #
  D_X = f_decoder_X(Z_given_X_samples)
  D_Y = f_decoder_Y(Z_given_Y_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stddev_explained, W_stddev_total, NLLK_X = \
  parse_output_distribution(
      X=T, D=D_X,
      dist_name=configs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stddev_explained, T_stddev_total, NLLK_Y = \
  parse_output_distribution(
      X=y, D=D_Y,
      dist_name=configs['ydist'], name='pY_given_Z')
  # final negative log likelihood
  NLLK = NLLK_X + get_masked_supervision(NLLK_Y, mask, nsample, configs['y_weight'])
  # ====== elbo ====== #
  kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
  loss = tf.reduce_mean(
      tf.reduce_logsumexp(NLLK + kl_weight * KL_X + KL_Y, axis=0))
  outputs = {'loss': tf.identity(loss, "loss"),
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = qZ_given_X.mean()
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stddev'] = W_stddev_total
  outputs['y'] = tf.reduce_mean(pY_given_Z.mean(), axis=0)
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

# ===========================================================================
# Semi-supervised latent VAE
# ===========================================================================
@N.Lambda
def oldmlvae(X, T, L, L_mean, L_var, mask, y, nsample, nepoch, configs):
  """ Multi-latent VAE """
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Multi_latent_VAE_X")
  # we only take the decoder here
  _, f_decoder_Y = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Multi_latent_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)

  (Z_X, qZ_given_X, Z_given_X_samples, KL_X) = \
  parse_latent_distribution(
      E=E_X,
      zdim=configs['zdim'] // 2, dist_name=configs['zdist'], name='Z1_Xlatent',
      n_samples=nsample,
      analytic=configs['analytic'])

  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=E_X,
      zdim=configs['zdim'] // 2, dist_name=configs['zdist'], name='Z2_Ylatent',
      n_samples=nsample,
      analytic=configs['analytic'])
  # ====== reconstruction and distortion ====== #
  # Z_given_XY_samples = tf.concat((Z_given_X_samples, Z_given_Y_samples),
  #                                axis=-1)
  D_X = f_decoder_X(Z_given_X_samples)
  D_Y = f_decoder_Y(Z_given_Y_samples)
  # for X -> W
  pX_given_Z, W_expected, W_stddev_explained, W_stddev_total, NLLK_X = \
  parse_output_distribution(
      X=T, D=D_X,
      dist_name=configs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stddev_explained, T_stddev_total, NLLK_Y = \
  parse_output_distribution(
      X=y, D=D_Y,
      dist_name=configs['ydist'], name='pY_given_Z')
  # final negative log likelihood
  NLLK = NLLK_X + get_masked_supervision(NLLK_Y, mask, nsample, configs['y_weight'])
  # ====== elbo ====== #
  kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
  loss = tf.reduce_mean(
      tf.reduce_logsumexp(NLLK + kl_weight * KL_X + KL_Y, axis=0))
  outputs = {'loss': loss,
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = tf.concat((qZ_given_X.mean(), qZ_given_Y.mean()),
                           axis=-1)
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stddev'] = W_stddev_total
  outputs['y'] = tf.reduce_mean(pY_given_Z.mean(), axis=0)
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

@N.Lambda
def olddlvae(X, T, L, L_mean, L_var, mask, y, nsample, nepoch, configs):
  """ Parallel latents VAE """
  # ====== create network ====== #
  f_encoder_X, f_decoder_X = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Dual_latent_VAE_X")
  f_encoder_Y, f_decoder_Y = create_network(
      hdim=configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], edrop=configs['edrop'],
      zdrop=configs['zdrop'], ddrop=configs['ddrop'],
      name="Dual_latent_VAE_Y")
  # ====== posterior ====== #
  E_X = f_encoder_X(X)
  E_Y = f_encoder_Y(y)

  (Z_X1, qZ_given_X1, Z_given_X1_samples, KL_X1) = \
  parse_latent_distribution(
      E=E_X,
      zdim=configs['zdim'] // 2, dist_name=configs['zdist'], name='Z1_Xlatent',
      n_samples=nsample,
      analytic=configs['analytic'])

  (Z_X2, qZ_given_X2, Z_given_X2_samples, KL_X2),\
  (Z_Y, qZ_given_Y, Z_given_Y_samples, KL_Y) = \
  parse_latent_distribution(
      E=[E_X, E_Y],
      zdim=configs['zdim'] // 2, dist_name=configs['zdist'], name='Z2_Ylatent',
      n_samples=nsample,
      analytic=configs['analytic'])

  # ====== reconstruction and distortion ====== #
  Z_given_XY_samples = tf.concat((Z_given_X1_samples, Z_given_X2_samples),
                                 axis=-1)
  D_X = f_decoder_X(Z_given_XY_samples)
  D_Y = f_decoder_Y(Z_given_Y_samples)

  # for X -> W
  pX_given_Z, W_expected, W_stddev_explained, W_stddev_total, NLLK_X = \
  parse_output_distribution(
      X=T, D=D_X,
      dist_name=configs['xdist'], name='pX_given_Z')
  # for y -> t
  pY_given_Z, T_expected, T_stddev_explained, T_stddev_total, NLLK_Y = \
  parse_output_distribution(
      X=y, D=D_Y,
      dist_name=configs['ydist'], name='pY_given_Z')
  # final negative log likelihood
  NLLK = NLLK_X + get_masked_supervision(NLLK_Y, mask, nsample, configs['y_weight'])
  # ====== elbo ====== #
  kl_weight = get_kl_weight(nepoch, configs['kl_weight'], configs['warmup'])
  loss = tf.reduce_mean(
      tf.reduce_logsumexp(NLLK + kl_weight * (KL_X1 + KL_X2) + KL_Y, axis=0))
  outputs = {'loss': loss,
             'metr': [tf.reduce_mean(NLLK_X, name="NLLK_X"),
                      tf.reduce_mean(NLLK_Y, name="NLLK_Y"),
                      tf.reduce_mean(KL_X1 + KL_X2, name="KLqp_X"),
                      tf.reduce_mean(KL_Y, name="KLqp_Y")]}
  # ====== latent mode ====== #
  outputs['Z'] = tf.concat((qZ_given_X1.mean(), qZ_given_X2.mean()),
                           axis=-1)
  # ====== output mode ====== #
  outputs['W'] = W_expected
  outputs['W_stddev'] = W_stddev_total
  outputs['y'] = tf.reduce_mean(pY_given_Z.mean(), axis=0)
  extract_pi(outputs, dist=pX_given_Z)
  extract_clean_count(outputs, dist=pX_given_Z)
  return outputs

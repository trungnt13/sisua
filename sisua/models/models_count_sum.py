from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

from odin import nnet as N, backend as K

from sisua.models.helpers import (parse_latent_distribution,
                                  parse_variable_distribution,
                                  create_network, get_count_sum,
                                  extract_pi)

# ===========================================================================
# Main models
# ===========================================================================
@N.Lambda
def cvae(X, C, mask, y, nsample, kwargs):
  """ M1 model:
  Generative: z -> x
  Inference : X -> z
  """
  print(X)
  print(C)
  exit()
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
      X=X, D=D,
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
  return outputs

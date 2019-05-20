"""
The following network design is used in this file:
#                  ____ y_prot
#                 /____ y_mRNA
#                /
# X ---- z ---- D -----X^
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from odin import nnet as N, backend as K

from sisua.models.helpers import parse_loss_function, create_network

# ===========================================================================
# Hard
# ===========================================================================
@N.Lambda
def multitask(X, T, C, mask, y, nsample, configs):
  # ====== prepare ====== #
  outputs = {}
  # ====== network ====== #
  f_encoder, f_decoder = create_network(
      configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], zdrop=configs['zdrop'],
      edrop=configs['edrop'], ddrop=configs['ddrop'],
      name='multitask')
  f_latent = N.Dense(num_units=configs['zdim'], b_init=None,
                     activation=K.linear, name="Bottleneck")
  # ====== output network ====== #
  f_x = N.Dense(num_units=X.shape.as_list()[-1], activation=K.linear,
                name="ReconstructionNetwork")
  f_y = N.Dense(num_units=y.shape.as_list()[-1], activation=K.linear,
                name="ProteinNetwork")
  # ====== latent space ====== #
  E = f_encoder(X)
  Z = f_latent(E)
  D = f_decoder(Z)
  # ====== reconstruction ====== #
  W, loss_rec = parse_loss_function(true=T, logit=f_x(D), mask=None,
                                    loss_name=configs['rec_loss'])
  # ====== classification ====== #
  y_, loss_cls = parse_loss_function(true=y, logit=f_y(D), mask=mask,
                                     loss_name=configs['cls_loss'])
  # ====== final output ====== #
  loss = loss_rec + loss_cls
  outputs.update({'W': W, 'Z': Z, 'y_': y,
                  'loss': tf.identity(loss, "Multitask_Loss"),
                  'metr': [tf.identity(loss_rec, "ReconstructionLoss"),
                           tf.identity(loss_cls, "SupervisedLoss")]})
  return outputs

@N.Lambda
def dualtask(X, T, C, mask, y, nsample, configs):
  # ====== prepare ====== #
  outputs = {}
  # ====== network ====== #
  f_X_encoder, f_X_decoder = create_network(
      configs['hdim'], batchnorm=configs['batchnorm'],
      n_layers=configs['nlayer'],
      xdrop=configs['xdrop'], zdrop=configs['zdrop'],
      edrop=configs['edrop'], ddrop=configs['ddrop'],
      name='multitask_X')

  f_Y_encoder, f_Y_decoder = create_network(
      configs['hdim'], batchnorm=configs['batchnorm'],
      xdrop=configs['xdrop'], zdrop=configs['zdrop'], edrop=configs['edrop'],
      name='multitask_Y')

  f_latent = N.Dense(num_units=configs['zdim'], b_init=None,
                     activation=K.linear, name="Bottleneck")
  # ====== output network ====== #
  f_X = N.Dense(num_units=X.shape.as_list()[-1], activation=K.linear,
                name="ReconstructionNetwork")
  f_Y = N.Dense(num_units=y.shape.as_list()[-1], activation=K.linear,
                name="ProteinNetwork")
  # ====== reconstruction ====== #
  E_X = f_X_encoder(X)
  Z_X = f_latent(E_X)
  D_X = f_X_decoder(Z_X)
  W, loss_rec = parse_loss_function(true=T, logit=f_X(D_X), mask=None,
                                    loss_name=configs['rec_loss'])
  # ====== classification ====== #
  E_Y = f_Y_encoder(y)
  Z_Y = f_latent(E_Y)
  D_Y = f_Y_decoder(Z_Y)
  y_, loss_cls = parse_loss_function(true=y, logit=f_Y(D_Y), mask=mask,
                                     loss_name=configs['cls_loss'])
  # ====== final output ====== #
  loss = loss_rec + loss_cls
  outputs.update({'W': W, 'Z': Z_X, 'y_': y,
                  'loss': tf.identity(loss, "Multitask_Loss"),
                  'metr': [tf.identity(loss_rec, "ReconstructionLoss"),
                           tf.identity(loss_cls, "SupervisedLoss")]})
  return outputs

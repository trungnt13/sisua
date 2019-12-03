from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf

from odin.networks import Identity
from sisua.models.base import SingleCellModel
from sisua.models.modules import create_encoder_decoder, get_latent


class SCALE(SingleCellModel):
  r""" Tensorflow implementation of SCALE

   Author: Lei Xiong - https://github.com/jsxlei
   License: https://github.com/jsxlei/SCALE/blob/master/LICENSE

   Reference:
    Xiong, L., Xu, K., Tian, K., et al., 2019. SCALE method for single-cell
    ATAC-seq analysis via latent feature extraction. Nature Communications.
    https://www.nature.com/articles/s41467-019-12630-7
  """

  def __init__(self,
               outputs,
               zdim=32,
               zcomponents=8,
               hdim=64,
               nlayers=2,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               kernel=5,
               stride=2,
               **kwargs):
    kwargs['analytic'] = False
    super().__init__(outputs, **kwargs)
    self.encoder, self.decoder = create_encoder_decoder(
        input_dim=self.omic_outputs[0].dim, seed=self.seed, **locals())
    self.latent = get_latent('mixdiag', zdim, n_components=int(zcomponents))

  def _call(self, x, lmean, lvar, t, y, mask, training, n_mcmc):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latent(e, training=training, n_mcmc=n_mcmc)
    # decoding the latent
    d = self.decoder(qZ, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX, qZ

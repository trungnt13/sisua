from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf

from odin.networks import Identity
from sisua.models.base import SingleCellModel
from sisua.models.utils import NetworkConfig, RandomVariable


class SCALE(SingleCellModel):
  r""" Tensorflow implementation of SCALE -
    "Single-Cell ATAC-seq analysis via Latent feature Extraction"

  Author: Lei Xiong - https://github.com/jsxlei
  License: https://github.com/jsxlei/SCALE/blob/master/LICENSE

   Reference:
    Xiong, L., Xu, K., Tian, K., et al., 2019. SCALE method for single-cell
      ATAC-seq analysis via latent feature extraction. Nature Communications.
      https://www.nature.com/articles/s41467-019-12630-7
  """

  def __init__(self, outputs, latent_dim=10, latent_components=8, **kwargs):
    kwargs['analytic'] = False
    latents = kwargs.pop('latents', None)
    if hasattr(latents, 'dim'):
      latent_dim = latents.dim
    elif isinstance(latents, (tuple, list)):
      latent_dim = latents[0].dim
    # override the latent
    latents = RandomVariable(int(latent_dim),
                             'mixdiag',
                             n_components=int(latent_components))
    super().__init__(outputs, latents, **kwargs)

  def encode(self, x, lmean=None, lvar=None, y=None, training=None, n_mcmc=1):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latents[0](e, training=training, n_mcmc=n_mcmc)
    return qZ

  def decode(self, z, training=None):
    # decoding the latent
    d = self.decoder(z, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX


class SCALAR(SCALE):
  r""" SCALE with semi-supervised extension -
    "Single-Cell ATAC-seq analysis via Latent and ADT Recombination"

  ADT: (antibody-derived tags)
  """

  def __init__(self,
               rna_dim=None,
               adt_dim=None,
               latent_dim=10,
               is_adt_probability=False,
               alternative_nb=False,
               **kwargs):
    # ====== output space ====== #
    outputs = kwargs.pop('outputs', None)
    if outputs is None:
      rna = RandomVariable(dim=rna_dim,
                           posterior='zinbd' if alternative_nb else 'zinb',
                           name='RNA')
      adt = RandomVariable(dim=adt_dim,
                           posterior='onehot' if is_adt_probability else
                           ('nbd' if alternative_nb else 'nb'),
                           name='ADT')
      outputs = [rna, adt]
    super().__init__(outputs,
                     latent_dim=latent_dim,
                     latent_components=int(adt_dim),
                     **kwargs)

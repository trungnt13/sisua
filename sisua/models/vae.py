from __future__ import absolute_import, division, print_function

import inspect
from typing import List

import tensorflow as tf

from odin.bay import RandomVariable
from odin.networks import Identity, NetworkConfig
from sisua.models.single_cell_model import SingleCellModel


class VariationalAutoEncoder(SingleCellModel):
  r""" Variational Auto Encoder """


class SISUA(VariationalAutoEncoder):
  r""" SemI-SUpervised Autoencoder

    - Transcriptomic : zero-inflated negative binomial distribution
    - Proteomic : negative binomial or onehot-categorical distribution
    - Latent : multi-variate normal with diagonal covariance

  Arguments:
    rna_dim : Integer, number of input dimension for scRNA-seq.
    adt_dim : Integer, number of input dimension for ADT.
    is_adt_probability : Boolean, if True, use `Bernoulli` for modeling the ADT,
      otherwise, use `NegativeBinomial`.
    alternative_nb : Boolean, if True, use mean-dispersion parameterization
      for negative binomial distribution.
  """

  def __init__(self,
               rna_dim=None,
               adt_dim=None,
               latent_dim=10,
               is_adt_probability=False,
               alternative_nb=False,
               **kwargs):
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
    super().__init__(outputs, latent_dim=latent_dim, **kwargs)


class MISA(VariationalAutoEncoder):
  r""" MIxture Semi-supervised Autoencoder

  Arguments:
    rna_dim : Integer, number of input dimension for scRNA-seq.
    adt_dim : Integer, number of input dimension for ADT.
    mixture_gaussian : Boolean, if True, use GMM for modeling the ADT,
      otherwise, use mixture of `NegativeBinomial`.
    alternative_nb : Boolean, if True, use mean-dispersion parameterization
      for negative binomial distribution.
  """

  def __init__(self,
               rna_dim=None,
               adt_dim=None,
               latent_dim=10,
               mixture_gaussian=False,
               alternative_nb=False,
               **kwargs):
    outputs = kwargs.pop('outputs', None)
    if outputs is None:
      rna = RandomVariable(dim=rna_dim,
                           posterior='zinbd' if alternative_nb else 'zinb',
                           name='RNA')
      kw = dict(n_components=2)
      if not mixture_gaussian:
        kw['alternative'] = alternative_nb
      adt = RandomVariable(
          dim=adt_dim,
          posterior='mixgaussian' if mixture_gaussian else 'mixnb',
          name='ADT',
          **kw)
      outputs = [rna, adt]
    super().__init__(outputs, latent_dim=latent_dim, **kwargs)

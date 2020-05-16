from __future__ import absolute_import, division, print_function

import inspect
import warnings
from typing import List

import tensorflow as tf

from odin.bay.vi.autoencoder import MultitaskVAE
from sisua.models.single_cell_model import RandomVariable, SingleCellModel


class VariationalAutoEncoder(SingleCellModel):
  r""" Variational Auto Encoder """


class SISUA(MultitaskVAE, SingleCellModel):
  r""" Multi-task SemI-SUpervised Autoencoder

    - Transcriptomic : zero-inflated negative binomial distribution
    - Proteomic : negative binomial or onehot-categorical distribution
    - Latent : multi-variate normal with diagonal covariance

    The following RandomVariable configurations are used in the paper:

    ```
    RandomVariable(rna_dim, 'zinbd'/'zinb', projection=True, name='RNA')
    RandomVariable(adt_dim, 'onehot'/'nbd'/'nb', True, 'ADT')
    ```

  Arguments:
    rna_dim : Integer, number of input dimension for scRNA-seq.
    adt_dim : Integer, number of input dimension for ADT.
    is_adt_probability : Boolean, if True, use `Bernoulli` for modeling the ADT,
      otherwise, use `NegativeBinomial`.
    alternative_nb : Boolean, if True, use mean-dispersion parameterization
      for negative binomial distribution.

  Reference:
    Ngo Trong, T., Kramer, R., Mehtonen, J., González, G., Hautamäki,
      V., Heinäniemi, M., 2019. "SISUA: Semi-Supervised Generative Autoencoder
      for Single Cell Data". bioRxiv. https://doi.org/10.1101/631382

  """

  def __init__(self, outputs, labels, **kwargs):
    super().__init__(outputs=outputs,
                     labels=labels,
                     input_shape=tf.nest.flatten(outputs)[0].event_shape,
                     **kwargs)


class MISA(SISUA):
  r""" MIxture labels for Semi-supervised Autoencoder """

  def __init__(self,
               outputs,
               labels,
               n_components=2,
               zero_inflated=False,
               **kwargs):
    labels = tf.nest.flatten(labels)
    n_components = int(n_components)
    zero_inflated = bool(zero_inflated)
    for rv in labels:
      if 'n_components' not in rv.kwargs:
        rv.kwargs['n_components'] = n_components
      # discrete count
      if rv.is_discrete or rv.is_binary:
        if rv.posterior[:3] != 'mix':
          warnings.warn("MISA only support labels is a mixture distribution "
                        f", given: {rv.posterior}")
          rv.posterior = 'mixnb'
        if 'zero_inflated' not in rv.kwargs:
          rv.kwargs['zero_inflated'] = zero_inflated
      # continuous
      elif rv.posterior[:3] not in ('mix', 'mdn'):
        warnings.warn("MISA only support labels is a mixture distribution "
                      f", given: {rv.posterior}")
        rv.posterior = 'mixgaussian'
    super().__init__(outputs=outputs, labels=labels, **kwargs)

from __future__ import absolute_import, division, print_function

import inspect
import warnings
from typing import List

import tensorflow as tf

from odin.bay.vi.autoencoder import MultitaskVAE
from sisua.models.single_cell_model import RandomVariable, SingleCellModel

__all__ = ['VAE', 'SISUA', 'MISA']


class VAE(SingleCellModel):
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
  r""" MIxture labels for Semi-supervised Autoencoder

  Example:
  ```
  sco = get_dataset("8kly")
  train, test = sco.split()
  print(train)
  # train.corrupt()
  rna = RandomVariable(sco.get_dim('transcriptomic'), 'zinb', True,
                       'transcriptomic')
  adt = RandomVariable(sco.get_dim('proteomic'), 'mixtril', True, 'proteomic')
  ######## Test
  vae = MISA(rna, adt, n_components=2)
  print(vae)
  vae.fit(train.create_dataset(train.omics, labels_percent=0.1),
          valid=test.create_dataset(test.omics, labels_percent=1.0),
          learning_rate=1e-3,
          valid_freq=500,
          compile_graph=True,
          max_iter=25000)
  vae.plot_learning_curves('/tmp/tmp.png')
  ```
  """

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
      # modify the n_components
      if 'n_components' not in rv.kwargs:
        rv.kwargs['n_components'] = n_components
    super().__init__(outputs=outputs, labels=labels, **kwargs)

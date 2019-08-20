from __future__ import absolute_import, division, print_function

from typing import Iterable, List, Text, Union

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from odin.bay import Statistic
from odin.bay.distribution_alias import parse_distribution
from odin.networks import DistributionDense, Parallel
from odin.utils import as_tuple
from sisua.models.autoencoder import DeepCountAutoencoder
from sisua.models.base import SingleCellModel
from sisua.models.variational_autoencoder import VariationalAutoEncoder


class MultitaskAutoEncoder(DeepCountAutoencoder):
  """ Multitask Auto Encoder
  """

  def __init__(self,
               dispersion='full',
               loss=['zinb', 'nb'],
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               biased_latent=False,
               nlayers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    kw = dict(locals())
    del kw['self']
    del kw['__class__']
    del kw['kwargs']
    kw.update(kwargs)
    super(MultitaskAutoEncoder, self).__init__(**kw)


class MultiTaskVAE(VariationalAutoEncoder):
  """ Variational autoencoder with multitask learning extension
  """

  def __init__(self,
               dispersion='full',
               xdist=['zinb', 'nb'],
               zdist='normal',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               nlayers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    kw = dict(locals())
    del kw['self']
    del kw['__class__']
    del kw['kwargs']
    kw.update(kwargs)
    super(MultiTaskVAE, self).__init__(**kw)


class MultiLatentVAE(VariationalAutoEncoder):
  pass

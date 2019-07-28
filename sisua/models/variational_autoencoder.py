from __future__ import print_function, division, absolute_import

import tensorflow as tf

from sisua.models.base import (
  DenseNetwork, DistributionLayer, SingleCellModel)

class VariationalAutoencoder(SingleCellModel):

  def __init__(self, dispersion='gene-cell',
               xdist='zinb', zdist='normal',
               xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
               hdim=128, zdim=32, nlayer=2,
               batchnorm=True, analytic=True,
               kl_weight=1., warmup=400, y_weight=10.,
               name=None):
    super(VariationalAutoencoder, self).__init__(
      name=name)

  def get_losses_and_metrics(self):
    pass
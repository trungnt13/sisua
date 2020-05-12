from __future__ import absolute_import, division, print_function

from typing import Tuple

import numpy as np

from sisua.data._single_cell_visualizer import _OMICvisualizer
from sisua.data.const import OMIC

__all__ = ['SingleCellOMIC', 'OMIC']


class SingleCellOMIC(_OMICvisualizer):
  r""" An annotated data matrix for storing multiple type of OMICs.

  Different OMIC types are stored in `obsm`

  Arguments:
    X : a matrix of shape `[n_cells, n_rna]`, transcriptomic
    cell_name : 1-D array of cell identification.
    gene_name : 1-D array of gene/rna identification.
    dtype : specific dtype for `X`
    name : identity of the single-cell dataset
    kwargs: extra keyword arguments for `scanpy.AnnData`

  Attributes:
    pass

  Methods:
    pass
  """

  def copy(self, filename=None) -> 'SingleCellOMIC':
    r""" Full copy, optionally on disk. (this code is copied from
    `AnnData`, modification to return `SingleCellOMIC` instance.
    """
    self._record('copy', locals())
    anndata = super().copy(filename)
    anndata._name = self.name
    sco = self.__class__(anndata, asview=False)
    return sco

  def split(self,
            train_percent=0.8,
            copy=True,
            seed=1) -> Tuple['SingleCellOMIC', 'SingleCellOMIC']:
    r""" Spliting the data into training and test dataset

    Arguments:
      train_percent : `float` (default=0.8)
        the percent of data used for training, the rest is for testing
      copy : a Boolean. if True, copy the data before splitting.
      seed : `int` (default=8)
        the same seed will ensure the same partition of any `SingleCellOMIC`,
        as long as all the data has the same number of `SingleCellOMIC.nsamples`

    Returns:
      train : `SingleCellOMIC`
      test : `SingleCellOMIC`

    Example:
    >>> x_train, x_test = x.split()
    >>> y_train, y_test = y.split()
    >>> assert np.all(x_train.obs['cellid'] == y_train.obs['cellid'])
    >>> assert np.all(x_test.obs['cellid'] == y_test.obs['cellid'])
    >>> #
    >>> x_train_train, x_train_test = x_train.split()
    >>> assert np.all(x_train_train.obs['cellid'] ==
    >>>               y_train[x_train_train.indices].obs['cellid'])
    """
    self._record('split', locals())
    assert 0 < train_percent < 1
    ids = np.random.RandomState(seed=seed).permutation(
        self.n_obs).astype('int32')
    ntrain = int(train_percent * self.n_obs)
    train_ids = ids[:ntrain]
    test_ids = ids[ntrain:]
    om = self.copy() if copy else self
    train = om[train_ids]
    test = om[test_ids]
    return train, test

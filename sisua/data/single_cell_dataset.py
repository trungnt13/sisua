from __future__ import absolute_import, division, print_function

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

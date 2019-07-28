from __future__ import print_function, division, absolute_import

from enum import Flag, auto
from six import add_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np

from sisua.data.single_cell_dataset import SingleCellOMICS

# ===========================================================================
# Helpers
# ===========================================================================
class NormMethod(Flag):
  RAW = auto()
  SUM = auto()
  LOG = auto()
  SCALE = auto()
  PROB = auto()
  BIN = auto()

  def apply(self, X, target_sum=None, trained_embedding=None):
    if self == NormMethod.SUM:
      X.normalize(total_counts_per_cell=True, target_sum=target_sum,
                  inplace=True)
    elif self == NormMethod.LOG:
      X.normalize(log1p=True, inplace=True)
    elif self == NormMethod.SCALE:
      X.normalize(scale=True, inplace=True)
    elif self == NormMethod.PROB or self == NormMethod.BIN:
      X.probabilistic_embedding(pbe=trained_embedding)
      X.obsm["X_org"] = X._X
      if self == NormMethod.PROB:
        X._X = X.obsm['X_prob']
        X._name = X._name[:-3] + 'prob'
      else:
        X._X = X.obsm['X_bin']
        X._name = X._name[:-3] + 'bin'
    return X

class FilterMethod(Flag):
  NONE = auto()
  VARIABLE_GENE = auto()
  GENE_COUNT = auto()
  CELL_COUNT = auto()

  def apply(self, X,
            min_counts_per_gene=1, min_cells_per_gene=None,
            min_counts_per_cell=1, min_genes_per_cell=None,
            min_disp=0.5, flavor='seurat', n_top_genes=None,
            gene_subset=None):
    if gene_subset is None:
      if self == FilterMethod.VARIABLE_GENE:
        X.filter_highly_variable_genes(
          min_disp=min_disp, n_top_genes=n_top_genes, flavor=flavor,
          inplace=True)
      elif self == FilterMethod.GENE_COUNT:
        X.filter_genes(min_counts=min_counts_per_gene,
                       min_cells=min_cells_per_gene,
                       inplace=True)
    else:
      if X.shape[1] > len(gene_subset):
        gene_ids = gene_subset.index.astype('int')
        X.apply_indices(gene_ids, observation=False)

    if self == FilterMethod.CELL_COUNT:
      X.filter_cells(min_counts=min_counts_per_cell,
                     min_genes=min_genes_per_cell,
                     inplace=True)
    return X

# ===========================================================================
# Recipes
# ===========================================================================
class NormalizationRecipe(object):

  def __init__(self, norm_method=NormMethod.RAW,
              filter_method=FilterMethod.NONE,
              corruption_rate=0.25, corruption_dist='binomial',
              target_sum=None,
              min_counts_per_gene=1, min_cells_per_gene=None,
              min_counts_per_cell=1, min_genes_per_cell=None,
              min_disp=0.5, flavor='seurat', n_top_genes=None):
    super(NormalizationRecipe, self).__init__()
    kw = dict(locals())
    del kw['self']
    self.__dict__.update(kw)
    # validate some important type
    assert isinstance(self.norm_method, NormMethod), \
      'norm_method must be instance of NormMethod'
    assert isinstance(self.filter_method, FilterMethod), \
      'filter_method must be instance of FilterMethod'
    self._is_trained = False
    self._gene_subset = None
    self._trained_embedding = None

  @property
  def name(self):
    return self.__class__.__name__

  @property
  def is_trained(self):
    return self._is_trained

  def normalize(self, X, training=False):
    if not isinstance(X, SingleCellOMICS):
      X = SingleCellOMICS(X=X)

    if self.filter_method != FilterMethod.NONE:
      assert X.obs.shape[1] > 0 and X.var.shape[1] > 0, \
        "Row and Column index is required when filtering is used!"

    if training:
      X.corrupt(corruption_rate=self.corruption_rate,
                corruption_dist=self.corruption_dist,
                inplace=True)

    # normalization is done first
    for method in list(NormMethod):
      if method in self.norm_method:
        method.apply(X, self.target_sum,
                     trained_embedding=self._trained_embedding)

    # filtering is done first
    for method in list(FilterMethod):
      if method in self.filter_method:
        method.apply(X,
              self.min_counts_per_gene, self.min_cells_per_gene,
              self.min_counts_per_cell, self.min_genes_per_cell,
              self.min_disp, self.flavor, self.n_top_genes,
              gene_subset=self._gene_subset)

    # in training mode, store the statistics for later testing:
    if training  and not self._is_trained:
      self._is_trained = True
      # assum the first var column is always geneID
      self._gene_subset = X.var.iloc[:, 0]
      if 'pbe' in X.uns:
        self._trained_embedding = X.uns['pbe']
    return X

  def __str__(self):
    return self.name

class Seurat(NormalizationRecipe):

  def _normalize(self, X, training=False):
    import scanpy as sc
    # pp.filter_cells(adata, min_genes=200)
    # pp.filter_genes(adata, min_cells=3)
    # pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    # filter_result = filter_genes_dispersion(
    #     adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5, log=not log)
    # if plot:
    #     from ..plotting import _preprocessing as ppp  # should not import at the top of the file
    #     ppp.filter_genes_dispersion(filter_result, log=not log)
    # adata._inplace_subset_var(filter_result.gene_subset)  # filter genes
    # if log: pp.log1p(adata)
    # pp.scale(adata, max_value=10)
    # return adata if copy else None

class CellRanger(NormalizationRecipe):

  def _normalize(self, X, training=False):
    import scanpy as sc
    # sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    # sc.pp.normalize_per_cell(                # normalize with total UMI count per cell
    #      adata, key_n_counts='n_counts_all')
    # filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
    #     adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False)
    # adata = adata[:, filter_result.gene_subset]     # subset the genes
    # sc.pp.normalize_per_cell(adata)          # renormalize after filtering
    # if log: sc.pp.log1p(adata)               # log transform: adata.X = log(adata.X + 1)
    # sc.pp.scale(adata)                       # scale to unit variance and shift to zero mean

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from enum import Flag, auto

import numpy as np
import scanpy as sc
from six import add_metaclass

from sisua.data.single_cell_dataset import SingleCellOMIC


# ===========================================================================
# Helpers
# ===========================================================================
class Methods(Flag):
  RAW = auto()
  CELL_COUNT = auto()
  GENE_COUNT = auto()
  TOTAL_COUNTS = auto()
  GENE_DISP = auto()
  EXP = auto()
  LOG = auto()
  SCALE = auto()
  PROB = auto()
  BIN = auto()

  @property
  def is_filtering(self):
    """ Return `True` if the normalization method contain any filter for
    cells or genes """
    if any(i in self for i in (Methods.CELL_COUNT, Methods.GENE_COUNT,
                               Methods.GENE_DISP)):
      return True
    return False

  @property
  def is_probabilistic_embedding(self):
    return Methods.PROB in self or Methods.BIN in self

  def __len__(self):
    count = 0
    for i in list(Methods)[1:]:
      if i in self:
        count += 1
    return count

  def __call__(self, X, **kwargs):
    return self.apply(X, **kwargs)

  def __iter__(self):
    for method in list(Methods)[1:]:
      if method in self:
        yield method

  def apply(self,
            X,
            min_counts_per_gene=1,
            min_cells_per_gene=None,
            min_counts_per_cell=1,
            min_genes_per_cell=None,
            min_disp=0.5,
            flavor='seurat',
            n_top_genes=None,
            target_sum=None,
            max_value=None,
            gene_subset=None,
            trained_embedding=None):
    assert isinstance(X, SingleCellOMIC), \
      "X must be instance of sisua.data.SingleCellOMIC"
    is_logarithmized = False
    for method in self:
      # cell filtering
      if method == Methods.CELL_COUNT:
        X.filter_cells(min_counts=min_counts_per_cell,
                       min_genes=min_genes_per_cell,
                       inplace=True)

      # gene filtering (cell countsums normalization is included)
      if gene_subset is None:
        if method == Methods.GENE_COUNT:
          X.filter_genes(min_counts=min_counts_per_gene,
                         min_cells=min_cells_per_gene,
                         inplace=True)
        if method == Methods.TOTAL_COUNTS:
          X.normalize(total_counts=True, target_sum=target_sum, inplace=True)
        if method == Methods.GENE_DISP:
          # Expects logarithmized data.
          X.normalize(log1p=True, inplace=True)
          is_logarithmized = True
          X.filter_highly_variable_genes(min_disp=min_disp,
                                         n_top_genes=n_top_genes,
                                         flavor=flavor,
                                         inplace=True)
      else:
        if method == Methods.TOTAL_COUNTS:
          X.normalize(total_counts=True, target_sum=target_sum, inplace=True)
        if X.shape[1] > len(gene_subset):
          gene_ids = gene_subset.index.astype('int')
          X.apply_indices(gene_ids, observation=False)

      # highly variable genes
      if method == Methods.EXP:
        X.expm1(inplace=True)
      elif method == Methods.LOG:
        if not is_logarithmized:
          X.normalize(log1p=True, inplace=True)
      elif method == Methods.SCALE:
        X.normalize(scale=True, max_value=max_value, inplace=True)
      elif method == Methods.PROB or method == Methods.BIN:
        X.probabilistic_embedding(pbe=trained_embedding)
        X.obsm["X_org"] = X._X
        if method == Methods.PROB:
          X._X = X.obsm['X_prob']
          X._name = X._name[:-3] + 'prob'
        else:
          X._X = X.obsm['X_bin']
          X._name = X._name[:-3] + 'bin'
    return X


# ===========================================================================
# Recipes
# ===========================================================================
class NormalizationRecipe(object):
  """ Predefined normalization recipe

  Note
  ----
  It is recommended to enable corruption within this normalization to
  simulate actual conditions
  """

  def __init__(self,
               methods: Methods = Methods.RAW,
               corruption_rate=0.25,
               corruption_dist='binomial',
               target_sum=None,
               max_value=None,
               min_counts_per_gene=1,
               min_cells_per_gene=None,
               min_counts_per_cell=1,
               min_genes_per_cell=None,
               min_disp=0.5,
               flavor='seurat',
               n_top_genes=None):
    super(NormalizationRecipe, self).__init__()
    # pylint just could not detect self.__dict__.update(locals())
    # so we assign each argument one-by-one
    self.methods = methods
    self.corruption_rate = corruption_rate
    self.corruption_dist = corruption_dist
    self.target_sum = target_sum
    self.max_value = max_value
    self.min_counts_per_gene = min_counts_per_gene
    self.min_cells_per_gene = min_cells_per_gene
    self.min_counts_per_cell = min_counts_per_cell
    self.min_genes_per_cell = min_genes_per_cell
    self.min_disp = min_disp
    self.flavor = flavor
    self.n_top_genes = n_top_genes
    # validate some important type
    assert isinstance(self.methods, Methods), \
      'methods must be instance of sisua.data.normalization_recipes.Methods'
    # some statistics might be stored during training
    self._is_trained = False
    self._gene_subset = None
    self._trained_embedding = None

  @property
  def name(self):
    return self.__class__.__name__.lower()

  @property
  def is_trained(self):
    return self._is_trained

  def __call__(self, X, training=False, copy=True):
    return self.normalize(X, training=training, copy=copy)

  def normalize(self, X, training=False, copy=True):
    """
    X : SingleCellOMIC
    training : data is used for training or not (
      corruption is enable for training)
    copy : create a copy version of data for normalization
    """
    if isinstance(X, SingleCellOMIC):
      X = X.copy() if copy else X
    elif isinstance(X, sc.AnnData):
      X = SingleCellOMIC(X.copy() if copy else X)
    else:
      X = SingleCellOMIC(X=X)

    # data should be corrupted before doing any preprocessing to
    # match the actual condition
    if training and self.corruption_rate > 0:
      X.corrupt(corruption_rate=self.corruption_rate,
                corruption_dist=self.corruption_dist,
                inplace=True)

    X = self.methods.apply(X,
                           min_counts_per_gene=self.min_counts_per_gene,
                           min_cells_per_gene=self.min_cells_per_gene,
                           min_counts_per_cell=self.min_counts_per_cell,
                           min_genes_per_cell=self.min_genes_per_cell,
                           min_disp=self.min_disp,
                           flavor=self.flavor,
                           n_top_genes=self.n_top_genes,
                           target_sum=self.target_sum,
                           max_value=self.max_value,
                           gene_subset=self._gene_subset,
                           trained_embedding=self._trained_embedding)

    if training and not self.is_trained:
      self._is_trained = True
      # assum the first var column is always geneID
      self._gene_subset = X.var.iloc[:, 0]
      if 'pbe' in X.uns:
        self._trained_embedding = X.uns['pbe']
    return X

  def __str__(self):
    return self.name


class Seurat(NormalizationRecipe):
  """Normalization and filtering as of Seurat [Satija15]_.
  This uses a particular preprocessing.
  Expects non-logarithmized data.

  Note
  ----
  It is recommended to enable corruption within this normalization to
  simulate actual conditions
  """

  def __init__(self,
               corruption_rate=0.25,
               corruption_dist='binomial',
               min_cells=3,
               min_genes=200,
               log=True):
    super(Seurat,
          self).__init__(methods=Methods.CELL_COUNT | Methods.GENE_COUNT |
                         Methods.TOTAL_COUNTS | Methods.GENE_DISP |
                         (Methods.LOG if log else Methods.EXP) | Methods.SCALE,
                         target_sum=1e4,
                         max_value=10,
                         min_counts_per_gene=None,
                         min_cells_per_gene=min_cells,
                         min_counts_per_cell=None,
                         min_genes_per_cell=min_genes,
                         min_disp=0.5,
                         flavor='seurat',
                         n_top_genes=None,
                         corruption_rate=corruption_rate,
                         corruption_dist=corruption_dist)


class CellRanger(NormalizationRecipe):
  """Normalization and filtering as of [Zheng17]_.
  Reproduces the preprocessing of [Zheng17]_ - the Cell Ranger R Kit of 10x
  Genomics.
  Expects non-logarithmized data.

  The procedure is as follow

  References
  ----------
  pass

  Note
  ----
  It is recommended to enable corruption within this normalization to
  simulate actual conditions
  """

  def __init__(self,
               corruption_rate=0.25,
               corruption_dist='binomial',
               min_counts=1,
               n_top_genes=1000,
               log=True):
    super(CellRanger, self).__init__(methods=Methods.GENE_COUNT |
                                     Methods.TOTAL_COUNTS | Methods.GENE_DISP,
                                     target_sum=None,
                                     max_value=None,
                                     min_counts_per_gene=min_counts,
                                     min_cells_per_gene=None,
                                     min_counts_per_cell=None,
                                     min_genes_per_cell=None,
                                     min_disp=None,
                                     flavor='cell_ranger',
                                     n_top_genes=n_top_genes,
                                     corruption_rate=corruption_rate,
                                     corruption_dist=corruption_dist)
    self.log = bool(log)

  def normalize(self, X, training=False, copy=True):
    """
    X : SingleCellOMIC
    training : data is used for training or not (
      corruption is enable for training)
    copy : create a copy version of data for normalization
    """
    import scanpy as sc
    sc.pp.recipe_seurat
    X = super(CellRanger, self).normalize(X, training=training, copy=copy)
    # renormalize after filtering
    X.expm1()
    X.normalize(total_counts=True, log1p=self.log, scale=True)
    return X


class Sisua(NormalizationRecipe):
  """
  Expects non-logarithmized data.
  """

  def __init__(self,
               corruption_rate=0.25,
               corruption_dist='binomial',
               min_counts=1,
               total_counts=True,
               n_top_genes=1000):
    methods = (Methods.CELL_COUNT | Methods.GENE_COUNT | Methods.TOTAL_COUNTS |
               (Methods.LOG if n_top_genes is None else Methods.GENE_DISP |
                Methods.LOG) | Methods.SCALE)
    if not total_counts:
      methods = methods ^ Methods.TOTAL_COUNTS
    super(Sisua, self).__init__(methods=methods,
                                target_sum=1e4,
                                max_value=None,
                                min_counts_per_gene=min_counts,
                                min_cells_per_gene=None,
                                min_counts_per_cell=min_counts,
                                min_genes_per_cell=None,
                                min_disp=0.5,
                                flavor='cell_ranger',
                                n_top_genes=n_top_genes,
                                corruption_rate=corruption_rate,
                                corruption_dist=corruption_dist)

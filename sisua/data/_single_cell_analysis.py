from __future__ import absolute_import, division, print_function

import inspect
import itertools
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import tensorflow as tf
from anndata._core.aligned_mapping import AxisArrays
from bigarray import MmapArrayWriter
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from six import string_types
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.decomposition import IncrementalPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from sklearn.mixture import GaussianMixture

from odin import visual as vs
from odin.search import diagonal_linear_assignment
from odin.stats import (describe, is_discrete, sparsity_percentage,
                        train_valid_test_split)
from odin.utils import (MPI, IndexedList, as_tuple, batching, cache_memory,
                        catch_warnings_ignore, cpu_count, is_primitive)
from odin.utils.crypto import md5_checksum
from sisua.data._single_cell_base import BATCH_SIZE, _OMICbase
from sisua.data.const import MARKER_ADT_GENE, MARKER_ADTS, MARKER_GENES, OMIC
from sisua.data.utils import (apply_artificial_corruption, get_library_size,
                              is_binary_dtype, is_categorical_dtype,
                              standardize_protein_name)
from sisua.label_threshold import ProbabilisticEmbedding


# ===========================================================================
# Helper
# ===========================================================================
def _threshold(x, nmin=2, nmax=5):
  if x.ndim == 1:
    x = x[:, np.newaxis]
  models = []
  aic = []
  for n_components in range(int(nmin), int(nmax)):
    gmm = GaussianMixture(n_components=n_components, random_state=1)
    gmm.fit(x)
    models.append(gmm)
    aic.append(gmm.aic(x))
  # select the best model
  gmm = models[np.argmax(aic)]
  y = gmm.predict(x)
  idx = np.argmax(gmm.means_.ravel())
  return (y == idx).astype(np.bool)


# ===========================================================================
# Main
# ===========================================================================
class _OMICanalyzer(_OMICbase):

  def get_x_probs(self, omic=None):
    r""" Return the probability embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[1]

  def get_x_bins(self, omic=None):
    r""" Return the binary embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[2]

  # ******************** transformation ******************** #
  def corrupt(self,
              omic=None,
              dropout_rate=0.2,
              retain_rate=0.2,
              distribution='binomial',
              inplace=True,
              seed=1):
    r"""
      omic : `OMIC`, which omic type will be corrupted
      dropout_rate : scalar (0.0 - 1.0), (default=0.25)
        how many entries (in percent) be selected for corruption.
      retain_rate : scalar (0.0 - 1.0), (default=0.2)
        how much percent of counts retained their original values.
      distribution : {'binomial', 'uniform} (default='binomial')
      omic : `sisua.data.OMIC`, which OMIC type will be corrupted
      inplace : `bool` (default=True). Perform computation inplace or return
        new `SingleCellOMIC` with the corrupted data.
      seed : `int` (default=8). Seed for the random state.
    """
    if omic is None:
      omic = self.current_omic
    om = self if inplace else self.copy()
    om._record('corrupt', locals())
    if not (0. < retain_rate < 1. or 0. < dropout_rate < 1.):
      return om
    for o in omic:
      apply_artificial_corruption(om.numpy(o),
                                  dropout=dropout_rate,
                                  retain_rate=retain_rate,
                                  distribution=distribution,
                                  copy=False,
                                  seed=seed)
      om._calculate_statistics(o)
    return om

  def filter_highly_variable_genes(self,
                                   min_disp: float = 1.0,
                                   max_disp: float = np.inf,
                                   min_mean: float = 0.01,
                                   max_mean: float = 8,
                                   n_top_genes: int = 1000,
                                   n_bins: int = 20,
                                   flavor: str = 'seurat',
                                   inplace: bool = True):
    r""" Annotate highly variable genes [Satija15]_ [Zheng17]_.

    https://www.rdocumentation.org/packages/Seurat/versions/2.3.4/topics/FindVariableGenes

    `Expects logarithmized data`.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    [Satija15]_ and Cell Ranger [Zheng17]_.

    The normalized dispersion is obtained by scaling with the mean and standard
    deviation of the dispersions for genes falling into a given bin for mean
    expression of genes. This means that for each bin of mean expression, highly
    variable genes are selected.

    Arguments:
      min_disp : `float`, optional (default=0.5)
          If `n_top_genes` unequals `None`, this and all other cutoffs for
          the means and the normalized dispersions are ignored.
      max_disp : `float`, optional (default=`np.inf`)
          If `n_top_genes` unequals `None`, this and all other cutoffs for
          the means and the normalized dispersions are ignored.
      min_mean : `float`, optional (default=0.0125)
          If `n_top_genes` unequals `None`, this and all other cutoffs for
          the means and the normalized dispersions are ignored.
      max_mean : `float`, optional (default=3)
          If `n_top_genes` unequals `None`, this and all other cutoffs for
          the means and the normalized dispersions are ignored.
      n_top_genes : {`float`, int`, `None`}, optional (default=`None`)
          Number of highly-variable genes to keep., if the value is in (0, 1],
          intepret as percent of genes
      n_bins : `int`, optional (default: 20)
          Number of bins for binning the mean gene expression. Normalization is
          done with respect to each bin. If just a single gene falls into a bin,
          the normalized dispersion is artificially set to 1.
      flavor : `{'seurat', 'cell_ranger'}`, optional (default='seurat')
          Choose the flavor for computing normalized dispersion. In their default
          workflows, Seurat passes the cutoffs whereas Cell Ranger passes
          `n_top_genes`.
      inplace : `bool` (default=True)
          if False, copy the `SingleCellOMIC` and apply the vargene filter.

    Returns:
      New `SingleCellOMIC` with filtered features if `applying_filter=True`
        else assign `SingleCellOMIC.highly_variable_features` with following
        attributes.

      highly_variable : bool
          boolean indicator of highly-variable genes
      **means**
          means per gene
      **dispersions**
          dispersions per gene
      **dispersions_norm**
          normalized dispersions per gene

    Notes:
      Proxy to `scanpy.pp.highly_variable_genes`. It is recommended to do
      `log1p` normalization before if `flavor='seurat'`.
    """
    flavor = str(flavor).lower()
    if n_top_genes is not None:
      if 0. < n_top_genes < 1.:
        n_top_genes = int(n_top_genes * self.n_vars)
    # prepare the data
    # this function will take the exponential of X all the time,
    # so non-logarithmzed data might led to overflow
    omics = self if inplace else self.copy()
    omics._record('filter_highly_variable_genes', locals())
    sc.pp.highly_variable_genes(omics,
                                min_disp=min_disp,
                                max_disp=max_disp,
                                min_mean=min_mean,
                                max_mean=max_mean,
                                n_top_genes=n_top_genes,
                                n_bins=int(n_bins),
                                flavor=flavor,
                                subset=True,
                                inplace=False)
    omics._name += '_vargene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def filter_genes(self,
                   min_counts=None,
                   max_counts=None,
                   min_cells=None,
                   max_cells=None,
                   inplace=True):
    r""" Filter features (columns) based on number of rows or counts.

    Keep columns that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_row_counts, max_row_counts]``

    Arguments:
      min_counts : {int, None} (default=None)
        Minimum number of counts required for a gene to pass filtering.
      max_counts : {int, None} (default=None)
        Maximum number of counts required for a gene to pass filtering.
      min_cells : {int, None} (default=None)
        Minimum number of cells expressed required for a feature to pass filtering.
      max_cells : {int, None} (default=None)
        Maximum number of cells expressed required for a feature to pass filtering.
      inplace : `bool` (default=True)
        if False, return new `SingleCellOMIC` with the filtered
        genes applied

    Returns:
      if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
      return new `SingleCellOMIC` with the new subset of genes

      gene_subset : `numpy.ndarray`
          Boolean index mask that does filtering. `True` means that the
          gene is kept. `False` means the gene is removed.
      number_per_gene : `numpy.ndarray`
          Depending on what was thresholded (`counts` or `cells`), the array
          stores `n_counts` or `n_cells` per gene.

    Note:
      Proxy method to Scanpy preprocessing
    """
    omics = self if inplace else self.copy()
    omics._record('filter_genes', locals())
    sc.pp.filter_genes(omics,
                       min_counts=min_counts,
                       max_counts=max_counts,
                       min_cells=min_cells,
                       max_cells=max_cells,
                       inplace=True)
    omics._name += '_filtergene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def filter_cells(self,
                   min_counts=None,
                   max_counts=None,
                   min_genes=None,
                   max_genes=None,
                   inplace=True):
    r""" Filter examples (rows) based on number of features or counts.

    Keep rows that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_col_counts, max_col_counts]``

    Arguments:
      min_counts : {int, None} (default=None)
        Minimum number of counts required for a cell to pass filtering.
      max_counts : {int, None} (default=None)
        Maximum number of counts required for a cell to pass filtering.
      min_genes : {int, None} (default=None)
        Minimum number of genes expressed required for a cell to pass filtering.
      max_genes : {int, None} (default=None)
        Maximum number of genes expressed required for a cell to pass filtering.
      inplace : `bool` (default=True)
        if False, return new `SingleCellOMIC` with the filtered
        cells applied

    Returns:
      if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
      return new `SingleCellOMIC` with the new subset of cells

      cells_subset : numpy.ndarray
          Boolean index mask that does filtering. ``True`` means that the
          cell is kept. ``False`` means the cell is removed.
      number_per_cell : numpy.ndarray
          Depending on what was tresholded (``counts`` or ``genes``), the array stores
          ``n_counts`` or ``n_cells`` per gene.

    Note:
      Proxy method to Scanpy preprocessing
    """
    # scanpy messed up here, the obs was not updated with the new indices
    cells_subset, number_per_cell = sc.pp.filter_cells(self,
                                                       min_counts=min_counts,
                                                       max_counts=max_counts,
                                                       min_genes=min_genes,
                                                       max_genes=max_genes,
                                                       inplace=False)
    omics = self if inplace else self.copy()
    omics._record('filter_cells', locals())
    omics.apply_indices(cells_subset, observation=True)
    omics._name += '_filtercell'
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def probabilistic_embedding(self,
                              omic=None,
                              n_components_per_class=2,
                              positive_component=1,
                              log_norm=True,
                              clip_quartile=0.,
                              remove_zeros=True,
                              ci_threshold=-0.68,
                              seed=1,
                              pbe: Optional[ProbabilisticEmbedding] = None):
    r""" Fit a GMM on each feature column to get the probability or binary
    representation of the features

    Return:
      `ProbabilisticEmbedding` model
      np.ndarray : probabilities X
      np.ndarray : binary X

    Arguments:
      pbe : {`sisua.ProbabilisticEmbedding`, `None`}, optional pretrained
        instance of `ProbabilisticEmbedding`
    """
    if omic is None:
      omic = self.current_omic
    self._record('probabilistic_embedding', locals())
    # We turn-off default log_norm here since the data can be normalized
    # separately in advance.
    omic = OMIC.parse(omic)
    X = self.numpy(omic)
    if X.shape[1] >= 100:
      warnings.warn("%d GMM will be trained!" % self.shape[1])
    name = omic.name
    pbe_name = '%s_pbe' % name
    prob_name = '%s_prob' % name
    bin_name = '%s_bin' % name
    label_name = self.get_labels_name(name)

    if is_binary_dtype(X):
      X_prob = X
      X_bin = X
      self.uns[pbe_name] = None
    else:
      if pbe is None:
        if pbe_name not in self.uns:
          pbe = ProbabilisticEmbedding(
              n_components_per_class=n_components_per_class,
              positive_component=positive_component,
              log_norm=log_norm,
              clip_quartile=clip_quartile,
              remove_zeros=remove_zeros,
              ci_threshold=ci_threshold,
              random_state=seed)
          with catch_warnings_ignore(ConvergenceWarning):
            pbe.fit(X)
          self.uns[pbe_name] = pbe
        else:
          pbe = self.uns[pbe_name]
      else:
        assert isinstance(pbe, ProbabilisticEmbedding), \
          'pbe, if given, must be instance of sisua.ProbabilisticEmbedding'
      # make prediction
      X_prob = np.clip(pbe.predict_proba(X), 0. + 1e-8, 1. - 1e-8)
      X_bin = pbe.predict(X)
    # store the data
    if prob_name not in self.obsm:
      self.obsm[prob_name] = X_prob
    if label_name not in self.obs and name + '_var' in self.uns:
      omic_id = self.get_var(name).index
      labels = [omic_id[i] for i in np.argmax(self.obsm[prob_name], axis=1)]
      self.obs[label_name] = pd.Categorical(labels)
    if bin_name not in self.obsm:
      self.obsm[bin_name] = X_bin
    return pbe, self.obsm[prob_name], self.obsm[bin_name]

  def dimension_reduce(self,
                       omic=None,
                       n_components=100,
                       algo='pca',
                       random_state=1):
    r""" Perform dimension reduction on given OMIC data. """
    if omic is None:
      omic = self.current_omic
    self._record('dimension_reduce', locals())
    algo = str(algo).lower().strip()
    assert algo in ('pca', 'tsne', 'umap'), \
      "Only support algorithm: 'pca', 'tsne', 'umap'; but given: '{algo}'"
    omic = OMIC.parse(omic)
    name = f"{omic.name}_{algo}"
    ## already transformed
    if name in self.obsm:
      return self.obsm[name] if n_components is None else \
        self.obsm[name][:, :int(n_components)]
    X = self.numpy(omic)
    n_components = min(n_components, X.shape[1])
    ### train new PCA model
    if algo == 'pca':
      X_ = np.empty(shape=(X.shape[0], n_components), dtype=X.dtype)
      model = IncrementalPCA(n_components=n_components)
      # fitting
      for start, end in batching(BATCH_SIZE, n=X.shape[0]):
        chunk = X[start:end]
        chunk = chunk.toarray() if issparse(chunk) else chunk
        model.partial_fit(chunk)
      # transforming
      for start, end in batching(BATCH_SIZE, n=X.shape[0]):
        chunk = X[start:end]
        chunk = chunk.toarray() if issparse(chunk) else chunk
        X_[start:end] = model.transform(chunk)
    ### TSNE
    elif algo == 'tsne':
      from odin.ml import fast_tsne
      X_ = fast_tsne(X, n_components=n_components, return_model=False)
      model = None
    ## UMAP
    elif algo == 'umap':
      try:
        import cuml
        method = 'rapids'
      except ImportError:
        method = 'umap'
      connectivities, distances, nn = self.neighbors(omic,
                                                     method='umap',
                                                     random_state=random_state)
      self.uns['neighbors'] = nn
      self.obsp['connectivities'] = connectivities
      self.obsp['distances'] = distances
      with catch_warnings_ignore(UserWarning):
        sc.tl.umap(self, method=method, random_state=random_state, copy=False)
      X_ = self.obsm['X_umap']
      model = self.uns['umap']
      del self.obsm['X_umap']
      del self.uns['umap']
      del self.uns['neighbors']
      del self.obsp['connectivities']
      del self.obsp['distances']
    ## store and return the result
    self.obsm[name] = X_
    # the model could be None, in case of t-SNE
    self.uns[name] = model
    return self.obsm[name] if n_components is None else \
      self.obsm[name][:, :int(n_components)]

  def expm1(self, omic=None, inplace=True):
    if omic is None:
      omic = self.current_omic
    om = self if inplace else self.copy()
    om._record('expm1', locals())
    _expm1 = lambda x: (np.expm1(x.data, out=x.data)
                        if issparse(x) else np.expm1(x, out=x))
    X = om.numpy(omic)
    for s, e in batching(n=self.n_obs, batch_size=BATCH_SIZE):
      X[s:e] = _expm1(X[s:e])
    om._calculate_statistics(omic)
    return om

  def normalize(self,
                omic=None,
                total=False,
                log1p=False,
                scale=False,
                target_sum=None,
                exclude_highly_expressed=False,
                max_fraction=0.05,
                max_value=None,
                inplace=True):
    r""" If ``exclude_highly_expressed=True``, very highly expressed genes are
    excluded from the computation of the normalization factor (size factor)
    for each cell. This is meaningful as these can strongly influence
    the resulting normalized values for all other genes [1]_.

    Arguments:
      total : bool (default=False). Normalize counts per cell.
      log1p : bool (default=False). Logarithmize the data matrix.
      scale : bool (default=False). Scale data to unit variance and zero mean.
      target_sum : {float, None} (default=None)
        If None, after normalization, each observation (cell) has a
        total count equal to the median of total counts for
        observations (cells) before normalization.
      exclude_highly_expressed : bool (default=False)
        Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than ``max_fraction`` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        ``target_sum``.
      max_fraction : bool (default=0.05)
        If ``exclude_highly_expressed=True``, consider cells as highly expressed
        that have more counts than ``max_fraction`` of the original total counts
        in at least one cell.
      max_value : `float` or `None`, optional (default=`None`)
          Clip (truncate) to this value after scaling. If `None`, do not clip.
      inplace : `bool` (default=True)
        if False, return new `SingleCellOMIC` with the filtered
        cells applied

    References:
      Weinreb et al. (2016), SPRING: a kinetic interface for visualizing
        high dimensional single-cell expression data, bioRxiv.

    Note:
      Proxy to `scanpy.pp.normalize_total`,  `scanpy.pp.log1p` and
        `scanpy.pp.scale`
    """
    if omic is None:
      omic = self.current_omic
    om = self if inplace else self.copy()
    om._record('normalize', locals())
    if omic != OMIC.transcriptomic:
      org_X = om._X
      om._X = om.numpy(omic)

    if total:
      sc.pp.normalize_total(om,
                            target_sum=target_sum,
                            exclude_highly_expressed=exclude_highly_expressed,
                            max_fraction=max_fraction,
                            inplace=True)
      # since the total counts is normalized, store the old library size
      om._name += '_total'

    if log1p:
      sc.pp.log1p(om, chunked=True, chunk_size=BATCH_SIZE, copy=False)
      om._name += '_log1p'
      del om.uns['log1p']
    # scaling may result negative total counts
    if scale:
      sc.pp.scale(om, zero_center=True, max_value=max_value, copy=False)
      om._name += '_scale'

    if omic != OMIC.transcriptomic:
      om.obsm[omic.name] = om.X
      om._X = org_X
    om._calculate_statistics(omic)
    return om

  # ******************** metrics ******************** #
  def neighbors(self,
                omic=None,
                n_neighbors=12,
                n_pcs=100,
                knn=True,
                method='umap',
                metric='euclidean',
                random_state=1):
    r"""\
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Arguments:
      n_neighbors : `int` (default=12)
        The size of local neighborhood (in terms of number of neighboring data
         points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.
      n_pcs : {`int`, `None`} (default=None)
        Use this many PCs. If n_pcs==0 use .X if use_rep is None.
        if n_pcs==None, use obsm['X_pca'].
      use_rep : {`None`, ‘X’} or any key for .obsm, optional (default=None)
        Use the indicated representation. If None, the representation is
        chosen automatically: for .n_vars < 50, .X is used, otherwise
        ‘X_pca’ is used. If ‘X_pca’ is not present, it’s computed with
        default parameters.
      knn : `bool` (default=True)
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
      method : {{'umap', 'gauss', `rapids`}}  (default: `'umap'`)
        Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
        with adaptive width [Haghverdi16]_) for computing connectivities.
        Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU
        only).
      metric : {`str`, `callable`} (default='euclidean')
        A known metric’s name or a callable that returns a distance.

    Returns:
      returns neighbors object with the following:

      **[OMIC]_connectivities** : sparse matrix (dtype `float32`)
          Weighted adjacency matrix of the neighborhood graph of data
          points. Weights should be interpreted as connectivities.
      **[OMIC]_distances** : sparse matrix (dtype `float32`)
          Instead of decaying weights, this stores distances for each pair of
          neighbors.
      **[OMIC]_neighbors** : dictionary
          configuration and params of fitted k-NN.
    """
    if omic is None:
      omic = self.current_omic
    self._record('neighbors', locals())
    omic = OMIC.parse(omic)
    name = f"{omic.name}_neighbors"
    if name not in self.uns:
      omic_name = omic.name
      if self.get_dim(omic) > 100:
        self.dimension_reduce(omic, algo='pca', random_state=random_state)
        omic_name = omic.name + '_pca'
      with catch_warnings_ignore(Warning):
        obj = sc.pp.neighbors(self,
                              n_neighbors=n_neighbors,
                              knn=knn,
                              method=method,
                              metric=metric,
                              n_pcs=int(n_pcs),
                              use_rep=omic_name,
                              random_state=random_state,
                              copy=True)
      self.uns[name] = obj.uns['neighbors']
      self.obsp[f"{omic.name}_connectivities"] = obj.obsp['connectivities']
      self.obsp[f"{omic.name}_distances"] = obj.obsp['distances']
      del obj
    return (self.obsp[f"{omic.name}_connectivities"],
            self.obsp[f"{omic.name}_distances"], self.uns[name])

  def clustering(self,
                 omic=None,
                 n_clusters=None,
                 n_init='auto',
                 algo='kmeans',
                 matching_labels=True,
                 return_key=False,
                 random_state=1):
    r""" Perform clustering for given OMIC type, the cluster labels will be
    assigned to `obs` with key "{omic}_{algo}{n_clusters}"

    Arguments:
      algo : {'kmeans', 'knn', 'pca', 'tsne', 'umap'}.
        Clustering algorithm, in case algo in ('pca', 'tsne', 'umap'),
        perform dimension reduction before clustering.
      matching_labels : a Boolean. Matching OMIC var_names to appropriate
        clusters, only when `n_clusters` is string or OMIC type.
      return_key : a Boolean. If True, return the name of the labels
        stored in `.obs` instead of the labels array.
    """
    if omic is None:
      omic = self.current_omic
    self._record('clustering', locals())
    ## clustering algorithm
    algo = str(algo).strip().lower()
    ## input data
    omic = OMIC.parse(omic)
    cluster_omic = None
    if n_clusters is None:
      cluster_omic = omic
      n_clusters = self.get_dim(omic)
    elif isinstance(n_clusters, Number):
      n_clusters = int(n_clusters)
    else:
      cluster_omic = OMIC.parse(n_clusters)
      n_clusters = self.get_dim(cluster_omic)
    n_clusters = int(n_clusters)
    n_init = int(n_init) if isinstance(n_init, Number) else \
      int(n_clusters) * 3
    ## check if output already extracted
    output_name = f"{omic.name}_{algo}{n_clusters}"
    if output_name in self.obs:
      return output_name if return_key else self.obs[output_name]
    ## warning
    if n_clusters > 50:
      warnings.warn(
          f"Found omic type:{cluster_omic} with {n_clusters} clusters")
    ## fit KMeans
    if algo in ('pca', 'tsne', 'umap', 'kmeans'):
      if algo in ('pca', 'tsne', 'umap'):
        X = self.dimension_reduce(omic=omic, n_components=100, algo=algo)
      else:
        X = self.numpy(omic)
      model = MiniBatchKMeans(n_clusters=int(n_clusters),
                              max_iter=1000,
                              n_init=int(n_init),
                              compute_labels=False,
                              batch_size=BATCH_SIZE,
                              random_state=random_state)
      # better suffering the batch
      for s, e in batching(BATCH_SIZE, self.n_obs, seed=random_state):
        x = X[s:e]
        model.partial_fit(x)
      # make prediction
      labels = []
      for s, e in batching(BATCH_SIZE, self.n_obs):
        x = X[s:e]
        labels.append(model.predict(x))
      labels = np.concatenate(labels, axis=0)
    ## fit KNN
    elif algo == 'knn':
      connectivities, distances, nn = self.neighbors(omic)
      n_neighbors = min(nn['params']['n_neighbors'],
                        np.min(np.sum(connectivities > 0, axis=1)))
      model = SpectralClustering(n_clusters=n_clusters,
                                 random_state=random_state,
                                 n_init=n_init,
                                 affinity='precomputed_nearest_neighbors',
                                 n_neighbors=n_neighbors)
      labels = model.fit_predict(connectivities)
    else:
      raise NotImplementedError(algo)
    ## correlation matrix
    if cluster_omic is not None and matching_labels:
      _, X, _ = self.probabilistic_embedding(cluster_omic)
      # omic-cluster correlation matrix
      corr = np.empty(shape=(X.shape[1], n_clusters), dtype=np.float32)
      for i, x in enumerate(X.T):
        for lab in range(n_clusters):
          mask = labels == lab
          corr[i, lab] = np.sum(x[mask])
      ids = diagonal_linear_assignment(corr)
      varnames = self.get_var_names(cluster_omic)
      labels_to_omic = {lab: name for lab, name, in zip(ids, varnames)}
      labels = np.array([labels_to_omic[i] for i in labels])
    ## saving data and model
    self.obs[output_name] = pd.Categorical(labels)
    # self.uns[output_name] = model
    return output_name if return_key else labels

  def louvain(self,
              omic=None,
              resolution=None,
              restrict_to=None,
              adjacency=None,
              flavor='vtraag',
              directed=True,
              use_weights=False,
              partition_type=None,
              partition_kwargs={},
              random_state=1):
    r"""Cluster cells into subgroups [Blondel08]_ [Levine15]_ [Traag17]_.

    Cluster cells using the Louvain algorithm [Blondel08]_ in the implementation
    of [Traag17]_. The Louvain algorithm has been proposed for single-cell
    analysis by [Levine15]_.

    This requires having ran :func:`~scanpy.pp.neighbors` or
    `~scanpy.external.pp.bbknn` first,
    or explicitly passing a ``adjacency`` matrix.

    Arguments:
      resolution
          For the default flavor (``'vtraag'``), you can provide a resolution
          (higher resolution means finding more and smaller clusters),
          which defaults to 1.0. See “Time as a resolution parameter” in [Lambiotte09]_.
      restrict_to
          Restrict the clustering to the categories within the key for sample
          annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
      key_added
          Key under which to add the cluster labels. (default: ``'louvain'``)
      adjacency
          Sparse adjacency matrix of the graph, defaults to
          ``adata.uns['neighbors']['connectivities']``.
      flavor : {``'vtraag'``, ``'igraph'``}
          Choose between to packages for computing the clustering.
          ``'vtraag'`` is much more powerful, and the default.
      directed
          Interpret the ``adjacency`` matrix as directed graph?
      use_weights
          Use weights from knn graph.
      partition_type
          Type of partition to use.
          Only a valid argument if ``flavor`` is ``'vtraag'``.
      partition_kwargs
          Key word arguments to pass to partitioning,
          if ``vtraag`` method is being used.
      random_state : Change the initialization of the optimization.

    Return:
      array `[n_samples]` : louvain community indices
      array `[n_samples]` : decoded louvain community labels
    """
    if omic is None:
      omic = self.current_omic
    self._record('louvain', locals())
    try:
      import louvain
    except ImportError:
      raise ImportError("pip install louvain>=0.6 python-igraph")
    omic = OMIC.parse(omic)
    output_name = omic.name + '_louvain'
    if output_name not in self.obs:
      with catch_warnings_ignore(Warning):
        connectivities, distances, nn = self.neighbors(omic)
        self.uns["neighbors"] = nn
        self.obsp["connectivities"] = connectivities
        self.obsp["distances"] = distances
        sc.tl.louvain(self,
                      resolution=resolution,
                      random_state=random_state,
                      restrict_to=restrict_to,
                      key_added=output_name,
                      adjacency=adjacency,
                      flavor=flavor,
                      directed=directed,
                      use_weights=use_weights,
                      partition_type=partition_type,
                      partition_kwargs=partition_kwargs,
                      copy=False)
      del self.uns['neighbors']
      del self.obsp["connectivities"]
      del self.obsp["distances"]
      model = self.uns['louvain']
      del self.uns['louvain']
      self.uns[output_name] = model
    y = self.obs[output_name].to_numpy().astype(np.float32)
    ### decode louvain community into labels
    output_labels = f"{output_name}_labels"
    if output_labels not in self.obs:
      var_names = self.get_var_names(omic)
      # mapping community_index -> confident value for each variables
      confidence = defaultdict(float)
      for i, x in zip(y, self.get_x_probs(omic=omic)):
        confidence[int(i)] += x
      # thresholding the variables
      labels = {}
      for community, x in confidence.items():
        labels[community] = '_'.join(var_names[_threshold(x, 2, 5)])
      # store in obs
      self.obs[output_labels] = np.array([labels[i] for i in y])
    ### return
    y_labels = self.obs[output_labels].to_numpy()
    return y, y_labels

  # ******************** Genes metrics and ranking ******************** #
  def top_vars(self, n_vars=100, return_indices=False):
    r""" The genes that are highly variated, high dispersion, less dropout
    (i.e. smallest counts of zero-values), and appeared in most cells
    will be returned.

    Arguments:
      return_indices : a Boolean. If True, return the index of top genes,
        otherwise, return the genes' ID.
    """
    self.calculate_quality_metrics()
    fnorm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # prepare data
    n_cells = fnorm(self.var['n_cells'].values)
    zeros = fnorm(self.var['pct_dropout'].values)
    dispersion = fnorm(self.var['dispersions'].values)
    # higher is better TODO: check again what is the best strategy here
    rating = n_cells + (1. - zeros) + dispersion
    ids = np.argsort(rating)[::-1]
    # indexing the genes
    genes = np.arange(self.n_vars, dtype=np.int64) \
      if return_indices else self.gene_id.values
    genes = genes[ids][:n_vars]
    return genes

  def rank_vars_groups(self,
                       n_vars=100,
                       group_by=OMIC.proteomic,
                       clustering='kmeans',
                       method='logreg',
                       corr_method='benjamini-hochberg',
                       max_iter=1000,
                       reference='rest'):
    r""" Rank genes for characterizing groups.

    Arguments:
      method : {'t-test_overestim_var', 't-test', 'wilcoxon', 'logreg'}
        - 't-test_overestim_var' overestimates variance of each group,
        - 't-test' uses t-test,
        - 'wilcoxon' uses Wilcoxon rank-sum,
        - 'logreg' uses logistic regression.
      corr_method :  p-value correction method.
        Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
      max_iter : an Integer.
        Only used for `method='logred'`

    Return:
      the key to ranked groups in `.uns`
    """
    self._record('rank_vars_groups', locals())
    # group by is categorical variables in `obs`
    if str(group_by) in self.obs:
      pass
    else:  # search in obsm, then clustering it
      group_by = OMIC.parse(group_by)
      if clustering is not None:
        group_by = self.clustering(group_by,
                                   n_clusters=group_by,
                                   algo=clustering,
                                   return_key=True)
      else:
        self.probabilistic_embedding(group_by)
        group_by = self.get_labels_name(group_by)
    ## check already ranked
    key = f'{self.get_current_omic().name}_{group_by}_rank'
    if key not in self.uns:
      kw = {}
      if method == 'logreg':
        kw['max_iter'] = int(max_iter)
        kw['random_state'] = 1
        kw['solver'] = 'saga'
      sc.tl.rank_genes_groups(self,
                              groupby=group_by,
                              n_genes=int(n_vars),
                              use_raw=True,
                              method=method,
                              corr_method=corr_method,
                              reference=reference,
                              copy=False,
                              key_added=key,
                              **kw)
    return key

  def calculate_quality_metrics(self,
                                n_bins=20,
                                flavor='cell_ranger',
                                percent_top=None,
                                log1p=False):
    r"""\
    Calculate quality control metrics for both the observations and variable.
    Highly variable genes (i.e. variables) also calculated.

    Arguments:
      n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
      flavor
        Choose the flavor for computing normalized dispersion. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
      percent_top : a list of Integer. Which proportions of top genes to cover.
        If empty or None don’t calculate. Values are considered 1-indexed,
        percent_top=[50] finds cumulative proportion to the 50th most
        expressed gene.
      log1p : a Boolean. If True, perform log1p before calculating the quality
        metrics, then, expm1 after the calculation.

    Observation level metrics include:
      "n_[omic.name]". Number of genes with positive counts in a cell.
      "total_[omic.name]". Total number of counts for a cell.
      "pct_counts_in_top_50_[omic.name]". Cumulative percentage of counts for 50 most
        expressed genes in a cell.

    Variable level metrics include:
      "total". Sum of counts for a gene.
      "mean". Mean expression over all cells.
      "n_cells". Number of cells this expression is measured in.
      "pct_dropout". Percentage of cells this feature does not
        appear in.
      "highly_variable" : boolean indicator of highly-variable genes
      "dispersions" : dispersions per gene
      "dispersions_norm" : normalized dispersions per gene

    """
    self._record('calculate_quality_metrics', locals())
    cell_qc, gene_qc = sc.pp.calculate_qc_metrics(
        self,
        percent_top=as_tuple(percent_top, t=int)
        if percent_top is not None else None,
        inplace=False)
    name = self._current_omic_name
    # var quality
    self.var['n_cells'] = gene_qc['n_cells_by_counts']
    self.var['mean'] = gene_qc['mean_counts']
    self.var['total'] = gene_qc['total_counts']
    self.var['pct_dropout'] = gene_qc['pct_dropout_by_counts']
    ## cell quality
    self.obs['n_%s' % name] = cell_qc['n_genes_by_counts']
    self.obs['total_%s' % name] = cell_qc['total_counts']
    if percent_top is not None:
      for i in as_tuple(percent_top, t=int):
        self.obs['pct_counts_in_top_%d_%s' % (i, name)] = \
          cell_qc['pct_counts_in_top_%d_genes' % i]
    ## Expects logarithmized data.
    if log1p:
      sc.pp.log1p(self)
    ## highly_variable, means, dispersions, dispersions_norm
    results = sc.pp.highly_variable_genes(self,
                                          n_bins=min(int(n_bins),
                                                     self.X.shape[1]),
                                          flavor=flavor,
                                          subset=False,
                                          inplace=False)
    self.var['highly_variable'] = [i[0] for i in results]
    self.var['dispersions'] = [i[2] for i in results]
    ## de-log
    if log1p:
      X = self.X
      for s, e in batching(BATCH_SIZE, n=self.n_obs):
        x = X[s:e]
        if sp.sparse.issparse(x):
          np.expm1(x.data, out=x.data)
        else:
          np.expm1(x, out=x)
    return self

  # ******************** other metrics ******************** #
  @cache_memory
  def get_marker_pairs(self,
                       omic1=OMIC.proteomic,
                       omic2=None,
                       var_names1=MARKER_ADTS,
                       var_names2=None,
                       threshold=None,
                       n=10,
                       most_correlated=False,
                       remove_duplicated=True):
    r""" Return the most differentiated (or correlated) pairs within a
    single OMIC (in case `omic2=None`) or between 2 different OMICs.

    Arguments:
      threshold : a Scalar.
        The minimum correlation value to be selected (in case
        `most_correlated=True`), otherwise, the maximum correlation value.
        If None, set the value to infinity.
      n : an Integer.
        Number of pairs with smallest correlation to be selected.
        If None, there is no limitation.
      most_correlated : a Boolean (default: True)
        if True, return most correlated pairs, otherwise, most un-correlated
        pairs.
      remove_duplicated : a Boolean (default: True)
        if True, remove pairs with duplicated name for first OMIC (in case
        `omic1 != omic2`), remove any pairs with duplicated name for both OMICs
        (in case `omic1 == omic2`).

    Return:
      list of tuple (var1, var2) sorted in order of the most correlated or
        un-correlated.
    """
    is_same_omic = False
    if omic2 is None:
      omic2 = omic1
      is_same_omic = True
    ids1 = self.get_var_indices(omic1)
    ids2 = self.get_var_indices(omic2)
    # check var_names
    if var_names1 is None:
      var_names1 = self.get_var_names(omic1)
    var_ids1 = set(ids1[i] for i in var_names1 if i in ids1)
    if len(var_ids1) == 0:
      raise ValueError(
          f"No matching variables found from given var_names={var_names1}")
    # for the second OMIC
    if var_names2 is None:
      var_names2 = self.get_var_names(omic2)
    var_ids2 = set(ids2[i] for i in var_names2 if i in ids2)
    if len(var_ids2) == 0:
      raise ValueError(
          f"No matching variables found from given var_names={var_names2}")
    # filtering
    var_names1 = self.get_var_names(omic1)
    var_names2 = self.get_var_names(omic2)
    scores = defaultdict(float)
    for i1, i2, p, s in self.get_correlation(omic1=omic1, omic2=omic2):
      if i1 not in var_ids1 or i2 not in var_ids2:
        continue
      name1 = var_names1[i1]
      name2 = var_names2[i2]
      key = (name1, name2)
      if is_same_omic:
        if name1 == name2:
          continue
        key = tuple(sorted(key))
      x = p + s
      if np.isnan(x):
        x = 1.0
      scores[key] += x
    scores = sorted(scores.items(), key=lambda x: x[-1])
    # most correlated
    if most_correlated:
      scores = scores[::-1]
    # prepare filtering
    threshold = (-np.inf if most_correlated else np.inf) \
      if threshold is None else float(threshold)
    n = np.inf if n is None else int(n)
    fn = lambda x: (x / 2 > threshold) if most_correlated else \
        (x / 2 < threshold)
    # filtering
    pairs = []
    seen = {}
    while True:
      if len(scores) == 0 or len(pairs) >= n:
        break
      key, val = scores.pop(0)
      if remove_duplicated:
        if is_same_omic:
          if any(k in seen for k in key):
            continue
          seen[key[0]] = 1
          seen[key[1]] = 1
        else:
          if key[0] in seen:
            continue
          seen[key[0]] = 1
      pairs.append(key)
    return pairs

  @cache_memory
  def get_importance_matrix(self,
                            omic=OMIC.transcriptomic,
                            target_omic=OMIC.proteomic,
                            random_state=1):
    r""" Using Tree Classifier to estimate the importance of each
    `omic` for each `target_omic`.
    """
    from odin.bay.vi.metrics import representative_importance_matrix
    from odin.bay.vi.utils import discretizing
    random_state = int(1)
    omic1 = self.current_omic if omic is None else OMIC.parse(omic)
    omic2 = self.current_omic if target_omic is None else OMIC.parse(
        target_omic)
    assert omic1 != omic2, "Mutual information only for 2 different OMIC type"
    uns_key = f"importance_{omic.name}_{target_omic.name}"
    if uns_key in self.uns:
      return self.uns[uns_key]
    # prepare data
    X = self.numpy(omic1)
    y = self.numpy(omic2)
    if not is_discrete(y):
      y = discretizing(y, n_bins=10, strategy='quantile')
    # split the data 50:50 for train and test
    rand = np.random.RandomState(random_state)
    ids = rand.permutation(X.shape[0])
    train = ids[:int(0.75 * X.shape[0])]
    test = ids[int(0.75 * X.shape[0]):]
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    # calculate the importance matrix
    matrix, train_acc, test_acc = representative_importance_matrix(
        repr_train=X_train,
        factor_train=y_train,
        repr_test=X_test,
        factor_test=y_test,
        random_state=rand.randint(1e8))
    self.uns[uns_key] = matrix
    return matrix

  @cache_memory
  def get_mutual_information(self,
                             omic=OMIC.transcriptomic,
                             target_omic=OMIC.proteomic,
                             n_neighbors=3,
                             random_state=1):
    r""" Estimate mutual information using k-NN

    Return
      a Matrix of shape `(n_features_omic, n_features_target_omic)`
        estimation of mutual information between each feature in `omic` to
        eacho feature in `target_omic`
    """
    n_neighbors = int(n_neighbors)
    random_state = int(1)
    omic1 = self.current_omic if omic is None else OMIC.parse(omic)
    omic2 = self.current_omic if target_omic is None else OMIC.parse(
        target_omic)
    assert omic1 != omic2, "Mutual information only for 2 different OMIC type"
    uns_key = f"mutualinfo_{omic.name}_{target_omic.name}"
    if uns_key in self.uns:
      return self.uns[uns_key]
    ### prepare the data
    x1 = self.numpy(omic1)
    x2 = self.numpy(omic2)
    n_om1 = x1.shape[1]
    n_om2 = x2.shape[1]
    discrete_features = np.array([is_discrete(i) for i in x1.T])

    def _mi(i2):
      y = x2[:, i2]
      if is_discrete(y):
        fn = mutual_info_classif
      else:
        fn = mutual_info_regression
      return i2, fn(X=x1,
                    y=y,
                    discrete_features=discrete_features,
                    n_neighbors=n_neighbors,
                    random_state=random_state)

    mi_mat = np.empty(shape=(n_om1, n_om2), dtype=np.float64)
    for i2, mi in MPI(list(range(n_om2)),
                      func=_mi,
                      ncpu=max(1,
                               cpu_count() - 1),
                      batch=1):
      mi_mat[:, i2] = mi
    self.uns[uns_key] = mi_mat
    return mi_mat

  @cache_memory
  def get_correlation(self, omic1=OMIC.transcriptomic, omic2=OMIC.proteomic):
    r""" Calculate the correlation scores between two omic types
    (could be different or the same OMIC).

    Return:
      list of tuple contained 4 scalars:
        (omic1-idx, omic2-idx, pearson, spearman)
        sorted in order from high to low average correlation
    """
    omic1 = self.current_omic if omic1 is None else OMIC.parse(omic1)
    omic2 = self.current_omic if omic2 is None else OMIC.parse(omic2)
    uns_key = f"correlation_{omic1.name}_{omic2.name}"
    if uns_key in self.uns:
      return self.uns[uns_key]
    ### prepare the data
    x1 = self.numpy(omic1)
    x2 = self.numpy(omic2)
    n_om1 = x1.shape[1]
    n_om2 = x2.shape[1]

    def _corr(ids):
      results = []
      if not isinstance(ids[0], tuple):
        ids = [ids]
      for i1, i2 in ids:
        y1 = x1[:, i1]
        y2 = x2[:, i2]
        with catch_warnings_ignore(RuntimeWarning):
          p = pearsonr(y1, y2)[0]
          s = spearmanr(y1, y2, nan_policy='omit').correlation
          # remove all NaNs
          results.append((i1, i2, p, s))
      yield results

    ### multiprocessing
    jobs = list(itertools.product(range(n_om1), range(n_om2)))
    ncpu = max(1, cpu_count() - 1)
    results = []
    for res in MPI(jobs, func=_corr, ncpu=ncpu, batch=len(jobs) // ncpu):
      results += res
    ### sorted by decreasing order
    all_correlations = sorted(
        results,
        key=lambda scores: (scores[-2] + scores[-1]) / 2,
    )[::-1]
    self.uns[uns_key] = all_correlations
    return all_correlations

from __future__ import absolute_import, division, print_function

import warnings
from itertools import product
from numbers import Number

import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from six import string_types

from odin import search
from odin import visual as vs
from odin.bay.vi.utils import discretizing
from odin.stats import sparsity_percentage
from odin.utils import as_tuple, catch_warnings_ignore
from odin.visual import Visualizer, to_axis
from sisua.data._single_cell_analysis import _OMICanalyzer
from sisua.data.const import (MARKER_ADT_GENE, MARKER_ADTS, MARKER_ATAC,
                              MARKER_GENES, OMIC)
from sisua.data.utils import is_categorical_dtype


# ===========================================================================
# Helper
# ===========================================================================
def _process_omics(sco, omic, clustering=None, allow_none=False):
  r""" Return the name of the observation and the extracted observation """
  if allow_none and (omic is None):
    return None, None
  if isinstance(omic, OMIC):
    omic = omic.name
  else:
    try:
      omic = OMIC.parse(omic).name
    except ValueError:
      pass
  omic = str(omic)
  x = None
  ## the omic provided already in observation
  if omic in sco.obs:
    x = sco.obs[omic].to_numpy()
  ## processing of multi-dimensional OMIC for labeling and clustering
  elif omic in sco.omics:
    # binary classes
    if np.all(sco.total_counts(omic).ravel() == 1.):
      label_name = f"{omic}_labels"
      if label_name in sco.obs: # already stored labels
        x = sco.obs[label_name]
      else: # one-hot encoded to labels vector
        labels = sco.get_var_names(omic)
        x = np.array([labels[i] for i in np.argmax(sco.get_omic(omic), axis=-1)])
        sco.obs[label_name] = x
      omic = label_name
    # Use Louvain community detection
    elif isinstance(clustering, string_types):
      clustering = clustering.lower().strip()
      if 'louvain' in clustering:  # community detection
        _, x = sco.louvain(omic)
        omic = omic + '_louvain'
      else:  # clustering
        n_clusters = None
        if omic == 'transcriptomic':
          for om in (OMIC.proteomic, OMIC.celltype, OMIC.iproteomic,
                     OMIC.icelltype):
            if om in sco:
              n_clusters = om
              break
        omic = sco.clustering(omic,
                              n_clusters=n_clusters,
                              algo=clustering,
                              return_key=True)
        x = sco.obs[omic].to_numpy()
    # probabilistic embedding
    else:
      x = sco.numpy(omic)
      _, prob, _ = sco.probabilistic_embedding(omic)
      try:
        x = sco.labels(omic).to_numpy()
        omic = sco.get_labels_name(omic)
      except KeyError:  # no variable name, just use raw integer values
        x = np.argmax(prob, axis=1)
  ## Exception
  else:
    raise ValueError("No support for omic: '%s' and clustering: '%s'" %
                     (omic, str(clustering)))
  return omic, x


def _process_varnames(sco, input_omic, var_names):
  input_omic = OMIC.parse(input_omic)
  if isinstance(var_names, string_types) and var_names == 'auto':
    var_names = input_omic.markers
  original_varnames = var_names
  with sco._swap_omic(input_omic) as sco:
    # select top variables
    if isinstance(var_names, Number):
      var_names = sco.top_vars(n_vars=int(var_names))
    # provided markers
    elif var_names is None:
      if input_omic == OMIC.transcriptomic:
        markers = set(MARKER_GENES)
      elif input_omic == OMIC.proteomic:
        markers = set(MARKER_ADTS)
      elif input_omic == OMIC.chromatin:
        markers = set(MARKER_ATAC)
      else:  # just take all variables
        markers = set(sco.var_names)
      var_names = [i for i in sco.var_names if i in markers]
    # given list of specific var_names
    else:
      var_names = [
          i for i in as_tuple(var_names, t=string_types)
          if i in set(sco.var_names)
      ]
  # check all var names are exist
  assert len(var_names) > 0, \
    (f"Cannot find appropriate variables name for OMIC type {input_omic.name}"
     f" given var_names={original_varnames}")
  return input_omic, sorted(var_names)


def _validate_arguments(kw):
  r""" Validate the argument and return a descriptive enough title for the
  figure """
  X = OMIC.parse(kw.get('X'))
  group_by = kw.get('group_by')
  if group_by is not None:
    group_by = OMIC.parse(group_by).name
  else:
    group_by = 'none'
  rank_genes = kw.get('rank_genes')
  clustering = kw.get('clustering')
  log = kw.get('log')
  if rank_genes:
    assert X == OMIC.transcriptomic, \
      f"Only visualize transcriptomic in case of rank_genes>0, but given: {X.name}"
  title = '_'.join(i for i in [
      X.name, group_by,
      str(clustering), ('rank' if rank_genes else ''), ('log' if log else 'raw')
  ] if len(i) > 0)
  return title


def _check_proteomic(self):
  if not OMIC.proteomic in self.omics:
    raise ValueError(
        "Require proteomic for plotting the marker gene/protein, given: %s" %
        str(self.omics))


def _adjust(fig, title, pad=0.02):
  w, h = fig.get_figwidth(), fig.get_figheight()
  fig.set_size_inches(w=w, h=h + 5)
  if title is not None:
    fig.suptitle(title)
  with catch_warnings_ignore(UserWarning):
    fig.tight_layout(rect=[0.0, pad, 1.0, 1.0 - pad])


# ===========================================================================
# Main class
# ===========================================================================
class _OMICvisualizer(_OMICanalyzer, Visualizer):

  #### Scatter plot

  def plot_scatter(self,
                   X=OMIC.transcriptomic,
                   color_by=OMIC.proteomic,
                   marker_by=None,
                   clustering='kmeans',
                   legend=True,
                   dimension_reduction='tsne',
                   max_scatter_points=5000,
                   ax=None,
                   fig=None,
                   title='',
                   return_figure=False):
    r""" Scatter plot of dimension using binarized protein labels

    Arguments:
      X : instance of OMIC.
        which OMIC data used for coordinates
      color_by : instance of OMIC.
        which OMIC data will be used for coloring the points
      marker_by : instance of OMIC.
        which OMIC data will be used for selecting the marker type
        (e.g. dot, square, triangle ...)
      clustering : {'kmeans', 'knn', 'pca', 'tsne', 'umap', 'louvain'}.
        Clustering algorithm, in case algorithm in ('pca', 'tsne', 'umap'),
        perform dimension reduction before clustering.
        Note: clustering is only applied in case of continuous data.
      dimension_reduction : {'tsne', 'umap', 'pca', None}.
        Dimension reduction algorithm. If None, just take the first 2
        dimension
    """
    ax = vs.to_axis2D(ax, fig=fig)
    omic = OMIC.parse(X)
    omic_name = omic.name
    max_scatter_points = int(max_scatter_points)
    ## prepare data
    X = self.dimension_reduce(omic, n_components=2, algo=dimension_reduction)
    color_name, colors = _process_omics(self,
                                        color_by,
                                        clustering=clustering,
                                        allow_none=True)
    marker_name, markers = _process_omics(self,
                                          marker_by,
                                          clustering=clustering,
                                          allow_none=True)
    ## downsampling
    if max_scatter_points > 0:
      ids = np.random.permutation(X.shape[0])[:max_scatter_points]
      X = X[ids]
      if colors is not None:
        colors = colors[ids]
      if markers is not None:
        markers = markers[ids]
    n_points = X.shape[0]
    ## ploting
    kw = dict(color='b')
    if colors is not None:
      if is_categorical_dtype(colors):  # categorical values
        kw['color'] = colors
      else:  # integral values
        kw['val'] = colors
        kw['color'] = 'bwr'
    name = '_'.join(str(i) for i in [omic_name, color_name, marker_name])
    title = f"[{dimension_reduction}-{name}]{title}"
    vs.plot_scatter(X,
                    marker='.' if markers is None else markers,
                    size=88 if n_points < 1000 else (120000 / n_points),
                    alpha=0.8,
                    legend_enable=bool(legend),
                    grid=False,
                    ax=ax,
                    title=title,
                    **kw)
    fig = ax.get_figure()
    if return_figure:
      return fig
    self.add_figure(f"scatter_{name}_{str(dimension_reduction).lower()}", fig)
    return self

  #### Others plot
  def plot_stacked_violins(self,
                           X=OMIC.transcriptomic,
                           group_by=OMIC.proteomic,
                           groups=None,
                           var_names='auto',
                           clustering='kmeans',
                           rank_vars=0,
                           dendrogram=False,
                           standard_scale='var',
                           log=False,
                           swap_axes=False,
                           title='',
                           return_figure=False):
    r""" Stacked violin plot of OMIC variables

    The y-axis contains `var_names` (e.g. `X`) and the x-axis the
    `group_by` categories.

    Arguments:
      X : instance of OMIC.
        Main data on the x-axis
      group_by : instance of OMIC.
        Data on y-axis for grouping the variables
      groups : a String or list of String.
        which groups in `group_by` variables will be shown
      var_names : an Integer, String, list of String, or None.
        select which variables in `X` will be presented
      rank_vars : an Integer.
        If greater than 0, rank variables in `X` by characterizing groups
        in `group_by`.
    """
    desc = f"{_validate_arguments(locals())}"
    X, var_names = _process_varnames(self, X, var_names)
    group_by, _ = _process_omics(self,
                                 group_by,
                                 clustering=clustering,
                                 allow_none=True)
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              log=log,
              standard_scale=standard_scale)
    with self._swap_omic(X):
      if rank_vars > 0:
        if group_by is None:
          raise ValueError("group_by must be provided in case rank_vars > 0.")
        rank_vars = int(rank_vars)
        key = self.rank_vars_groups(group_by=group_by, n_vars=rank_vars)
        axes = sc.pl.rank_genes_groups_stacked_violin(self,
                                                      n_genes=rank_vars,
                                                      groups=groups,
                                                      key=key,
                                                      **kw)
      else:
        axes = sc.pl.stacked_violin(self,
                                    var_names=var_names,
                                    groupby=group_by,
                                    **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    plt.title(f"[{title}]{desc}")
    _adjust(fig, title=None, pad=0.01)
    if return_figure:
      return fig
    self.add_figure(f"violin_{desc}", fig)
    return self

  def plot_dotplot(self,
                   X=OMIC.transcriptomic,
                   group_by=OMIC.transcriptomic,
                   var_names='auto',
                   clustering='kmeans',
                   rank_genes=0,
                   dendrogram=False,
                   standard_scale='var',
                   cmap='Reds',
                   log=True):
    desc = _validate_arguments(locals())
    X, var_names = _process_varnames(self, X, var_names)
    group_by, _ = _process_omics(self,
                                 group_by,
                                 clustering=clustering,
                                 allow_none=True)
    kw = dict(dendrogram=dendrogram,
              log=log,
              standard_scale=standard_scale,
              color_map=cmap)
    ## ploting
    with self._swap_omic(X):
      if rank_genes > 0:
        if group_by is None:
          raise ValueError("group_by must be provided in case rank_vars > 0.")
        key = self.rank_vars_groups(groupby=group_by, n_vars=int(rank_genes))
        axes = sc.pl.rank_genes_groups_dotplot(self,
                                               n_genes=int(rank_genes),
                                               key=key,
                                               **kw)
      else:
        axes = sc.pl.dotplot(self, var_names=var_names, groupby=group_by, **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    _adjust(fig, desc)
    self.add_figure(f"dotplot_{desc}", fig)
    return self

  #### Heatmap plot
  def plot_heatmap(self,
                   X=OMIC.transcriptomic,
                   group_by=OMIC.proteomic,
                   groups=None,
                   var_names='auto',
                   clustering='kmeans',
                   rank_vars=0,
                   dendrogram=False,
                   swap_axes=False,
                   cmap='bwr',
                   standard_scale='var',
                   log=True,
                   title='',
                   return_figure=False):
    r""" Heatmap of the expression values of genes.

    If `group_by` is given, the heatmap is ordered by the respective group.

    Arguments:
      X : instance of OMIC.
        which OMIC data used for coordinates
      group_by : instance of OMIC.
        which OMIC data will be used to order the features
      clustering : {'kmeans', 'knn', 'pca', 'tsne', 'umap'}.
        Clustering algorithm, in case algo in ('pca', 'tsne', 'umap'),
        perform dimension reduction before clustering.
        Note: clustering is only applied in case of continuous data.
      groups : a String or list of String.
        which groups in `group_by` variables will be shown
      var_names : an Integer, String, list of String, or None.
        select which variables in `X` will be presented
      rank_vars : an Integer.
        If greater than 0, rank variables in `X` by characterizing groups
        in `group_by`.
    """
    desc = f"{_validate_arguments(locals())}"
    X, var_names = _process_varnames(self, X, var_names)
    group_by, _ = _process_omics(self,
                                 group_by,
                                 clustering=clustering,
                                 allow_none=True)
    ## select the right marker genes and omic
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              standard_scale=standard_scale,
              show_gene_labels=True if len(var_names) < 50 else False,
              log=log,
              cmap=cmap)
    ## plotting
    with self._swap_omic(X):
      if rank_vars > 0:
        if group_by is None:
          raise ValueError("group_by must be provided in case rank_vars > 0.")
        key = self.rank_vars_groups(group_by=group_by, n_vars=rank_vars)
        axes = sc.pl.rank_genes_groups_heatmap(self,
                                               n_genes=rank_vars,
                                               groups=groups,
                                               key=key,
                                               **kw)
      else:
        axes = sc.pl.heatmap(self, var_names=var_names, groupby=group_by, **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    fig.get_axes()[0].set_title(f"[{title}]{desc}")
    _adjust(fig, title=None)
    if return_figure:
      return fig
    self.add_figure(f'heatmap_{desc}', fig)
    return self

  def plot_distance_heatmap(self,
                            X=OMIC.transcriptomic,
                            group_by=OMIC.transcriptomic,
                            var_names='auto',
                            clustering='kmeans',
                            cmap='bwr',
                            legend=True,
                            log=True,
                            ax=None,
                            title='',
                            return_figure=False):
    r""" Heatmap of the distance among latents vector from different classes

    Arguments:
      X : instance of OMIC.
        which OMIC data used for coordinates
      group_by : instance of OMIC.
        which OMIC data will be used to order the features
      clustering : {'kmeans', 'knn', 'pca', 'tsne', 'umap'}.
        Clustering algorithm, in case algo in ('pca', 'tsne', 'umap'),
        perform dimension reduction before clustering.
        Note: clustering is only applied in case of continuous data.
    """
    title = f"[{_validate_arguments(locals())}]{title}"
    X, var_names = _process_varnames(self, X, var_names)
    group_by, _ = _process_omics(self,
                                 group_by,
                                 clustering=clustering,
                                 allow_none=False)
    ax = vs.to_axis2D(ax)
    ## prepare the data
    vs.plot_distance_heatmap(self.numpy(X),
                             labels=self.numpy(group_by),
                             colormap=cmap,
                             legend_enable=legend,
                             lognorm=log,
                             ax=ax,
                             fontsize=8,
                             legend_ncol=3,
                             title=title)
    fig = ax.get_figure()
    if return_figure:
      return fig
    self.add_figure(f'distance_heatmap_{title}', fig)
    return self

  #### Correlation plot

  def _plot_heatmap_matrix(self,
                           matrix,
                           figname,
                           omic1=OMIC.transcriptomic,
                           omic2=OMIC.proteomic,
                           var_names1=MARKER_ADT_GENE.values(),
                           var_names2=MARKER_ADT_GENE.keys(),
                           is_marker_pairs=True,
                           title='',
                           return_figure=False):
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    if isinstance(var_names1, string_types) and var_names1 == 'auto':
      var_names1 = omic1.markers
    if isinstance(var_names2, string_types) and var_names2 == 'auto':
      var_names2 = omic2.markers
    if var_names1 is None or var_names2 is None:
      is_marker_pairs = False
    names1 = self.get_var_names(omic1)
    names2 = self.get_var_names(omic2)
    om1_idx = {j: i for i, j in enumerate(names1)}
    om2_idx = {j: i for i, j in enumerate(names2)}
    assert matrix.shape == (len(names1), len(names2)), \
      (f"Given OMIC {omic1.name}({len(names1)} variables) and "
       f"OMIC {omic2.name}({len(names2)} variables) "
       f"mistmach matrix shape {matrix.shape}")
    ## filter the variables
    if is_marker_pairs:
      pairs = [(v1, v2)
               for v1, v2 in zip(var_names1, var_names2)
               if v1 in om1_idx and v2 in om2_idx]
      var_names1 = [i for i, _ in pairs]
      var_names2 = [i for _, i in pairs]
    if var_names1 is not None:
      names1 = np.array([i for i in var_names1 if i in om1_idx])
      matrix = matrix[[om1_idx[i] for i in names1]]
    if var_names2 is not None:
      names2 = np.array([i for i in var_names2 if i in om2_idx])
      matrix = matrix[:, [om2_idx[i] for i in names2]]
    ## find the best diagonal match
    ids2 = search.diagonal_linear_assignment(matrix, nan_policy=0)
    matrix = matrix[:, ids2]
    names2 = names2[ids2].tolist()
    names1 = names1.tolist()
    n1 = len(names1)
    n2 = len(names2)

    ## helper for marking the marker
    def _mark(ax):
      if is_marker_pairs:
        for v1, v2 in zip(var_names1, var_names2):
          x = names2.index(v2)
          y = names1.index(v1)
          ax.text(x + 0.02,
                  y + 0.03,
                  s='X',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=88 / np.log1p(max(n1, n2)),
                  color='magenta',
                  alpha=0.8,
                  weight='regular')

    ## plotting
    styles = dict(cmap="bwr",
                  xticklabels=names2,
                  yticklabels=names1,
                  xlabel=omic2.name,
                  ylabel=omic1.name,
                  gridline=0.1,
                  fontsize=10,
                  cbar=True)
    width = min(25, matrix.shape[1] / 1.2)
    fig = plt.figure(figsize=(width, width * matrix.shape[0] / matrix.shape[1]))
    _mark(
        vs.plot_heatmap(
            matrix,
            **styles,
            ax=None,
            title=f"[{figname}_x:{omic2.name}_y:{omic1.name}]{title}"))
    with catch_warnings_ignore(UserWarning):
      fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
    ## store and return
    if return_figure:
      return fig
    self.add_figure(f"{figname.lower()}_{omic1.name}_{omic2.name}", fig)
    return self

  def plot_importance_matrix(self,
                             omic1=OMIC.transcriptomic,
                             omic2=OMIC.proteomic,
                             var_names1=MARKER_ADT_GENE.values(),
                             var_names2=MARKER_ADT_GENE.keys(),
                             is_marker_pairs=True,
                             title='',
                             return_figure=False):
    r""" Showing importance heatmap matrix """
    return self._plot_heatmap_matrix(
        matrix=self.get_importance_matrix(omic1, omic2),
        figname="Importance",
        omic1=omic1,
        omic2=omic2,
        var_names1=var_names1,
        var_names2=var_names2,
        is_marker_pairs=is_marker_pairs,
        title=title,
        return_figure=return_figure,
    )

  def plot_mutual_information(self,
                              omic1=OMIC.transcriptomic,
                              omic2=OMIC.proteomic,
                              var_names1=MARKER_ADT_GENE.values(),
                              var_names2=MARKER_ADT_GENE.keys(),
                              is_marker_pairs=True,
                              title='',
                              return_figure=False):
    r""" Plot estimated mutual information between each variable pair in
    omic1 and omic2. """
    return self._plot_heatmap_matrix(
        matrix=self.get_mutual_information(omic1, omic2),
        figname="MutualInfo",
        omic1=omic1,
        omic2=omic2,
        var_names1=var_names1,
        var_names2=var_names2,
        is_marker_pairs=is_marker_pairs,
        title=title,
        return_figure=return_figure,
    )

  def plot_pearson_matrix(self,
                          omic1=OMIC.transcriptomic,
                          omic2=OMIC.proteomic,
                          var_names1=MARKER_ADT_GENE.values(),
                          var_names2=MARKER_ADT_GENE.keys(),
                          is_marker_pairs=True,
                          title='',
                          return_figure=False):
    r""" Plot correlation matrix between omic1 and omic2

    Arguments:
      omic1, omic2 : instance of OMIC.
        With `omic1` represent the x-axis, and `omic2` represent the y-axis.
      markers : a List of String (optional)
        a list of `omic1` variable that should be most coordinated to `omic2`
    """
    n_var1 = self.get_n_var(omic1)
    n_var2 = self.get_n_var(omic2)
    corr = self.get_correlation(omic1, omic2)
    pearson = np.zeros(shape=(n_var1, n_var2), dtype=np.float64)
    for i1, i2, p, s in corr:
      pearson[i1, i2] = p
    return self._plot_heatmap_matrix(
        matrix=pearson,
        figname="Pearson",
        omic1=omic1,
        omic2=omic2,
        var_names1=var_names1,
        var_names2=var_names2,
        is_marker_pairs=is_marker_pairs,
        title=title,
        return_figure=return_figure,
    )

  def plot_spearman_matrix(self,
                           omic1=OMIC.transcriptomic,
                           omic2=OMIC.proteomic,
                           var_names1=MARKER_ADT_GENE.values(),
                           var_names2=MARKER_ADT_GENE.keys(),
                           is_marker_pairs=True,
                           title='',
                           return_figure=False):
    r""" Plot correlation matrix between omic1 and omic2
    """
    n_var1 = self.get_n_var(omic1)
    n_var2 = self.get_n_var(omic2)
    corr = self.get_correlation(omic1, omic2)
    spearman = np.zeros(shape=(n_var1, n_var2), dtype=np.float64)
    for i1, i2, p, s in corr:
      spearman[i1, i2] = s
    return self._plot_heatmap_matrix(
        matrix=spearman,
        figname="Spearman",
        omic1=omic1,
        omic2=omic2,
        var_names1=var_names1,
        var_names2=var_names2,
        is_marker_pairs=is_marker_pairs,
        title=title,
        return_figure=return_figure,
    )

  def plot_correlation_scatter(self,
                               omic1=OMIC.transcriptomic,
                               omic2=OMIC.proteomic,
                               var_names1='auto',
                               var_names2='auto',
                               is_marker_pairs=True,
                               log1=True,
                               log2=True,
                               max_scatter_points=200,
                               top=3,
                               bottom=3,
                               title='',
                               return_figure=False):
    r""" Mapping from omic1 to omic2

    Arguments:
      omic1, omic2 : instance of OMIC.
        With `omic1` represent the x-axis, and `omic2` represent the y-axis.
      var_names1 : list of all variable name for `omic1`
    """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    if isinstance(var_names1, string_types) and var_names1 == 'auto':
      var_names1 = omic1.markers
    if isinstance(var_names2, string_types) and var_names2 == 'auto':
      var_names2 = omic2.markers
    if var_names1 is None or var_names2 is None:
      is_marker_pairs = False
    max_scatter_points = int(max_scatter_points)
    # get all correlations
    corr = self.get_correlation(omic1, omic2)
    corr_map = {(x[0], x[1]):
                (0 if np.isnan(x[2]) else x[2], 0 if np.isnan(x[3]) else x[3])
                for x in corr}
    om1_names = self.get_var_names(omic1)
    om2_names = self.get_var_names(omic2)
    om1_idx = {j: i for i, j in enumerate(om1_names)}
    om2_idx = {j: i for i, j in enumerate(om2_names)}
    # extract the data and normalization
    X1 = self.numpy(omic1)
    library = np.sum(X1, axis=1, keepdims=True)
    library = discretizing(library, n_bins=10, strategy='quantile').ravel()
    if log1:
      s = np.sum(X1, axis=1, keepdims=True)
      X1 = np.log1p(X1 / s * np.median(s))
    X2 = self.numpy(omic2)
    if log2:
      s = np.sum(X2, axis=1, keepdims=True)
      X2 = np.log1p(X2 / s * np.median(s))
    ### getting the marker pairs
    all_pairs = []
    # coordinate marker pairs
    if is_marker_pairs:
      pairs = [(i1, i2)
               for i1, i2 in zip(var_names1, var_names2)
               if i1 in om1_idx and i2 in om2_idx]
      var_names1 = [i for i, _ in pairs]
      var_names2 = [i for _, i in pairs]
    # filter omic2
    if var_names2 is not None:
      var_names2 = [i for i in var_names2 if i in om2_names]
    else:
      var_names2 = om2_names
    assert len(var_names2) > 0, \
      (f"None of the variables {var_names2} is contained in variable list "
       f"of OMIC {omic2.name}")
    nrow = len(var_names2)
    # filter omic1
    if var_names1 is not None:
      var_names1 = [i for i in var_names1 if i in om1_names]
      ncol = len(var_names1)
      assert len(var_names1) > 0, \
        (f"None of the variables {var_names1} is contained in variable list "
         f"of OMIC {omic1.name}")
      for name2 in var_names2:
        for name1 in var_names1:
          all_pairs.append((om1_idx[name1], om2_idx[name2]))
    else:
      # top and bottom correlation pairs
      top = int(top)
      bottom = int(bottom)
      ncol = top + bottom
      # pick all top and bottom of omic1 coordinated to omic2
      for name in var_names2:
        i2 = om2_idx[name]
        pairs = sorted(
            [[sum(corr_map[(i1, i2)]), i1] for i1 in range(len(om1_names))])
        for _, i1 in pairs[-top:][::-1] + pairs[:bottom][::-1]:
          all_pairs.append((i1, i2))
    ### downsampling scatter points
    if max_scatter_points > 0:
      ids = np.random.permutation(len(X1))[:max_scatter_points]
    else:
      ids = np.arange(len(X1), dtype=np.int32)
    ### plotting
    fig = plt.figure(figsize=(ncol * 2, nrow * 2 + 2), dpi=80)
    for i, pair in enumerate(all_pairs):
      ax = plt.subplot(nrow, ncol, i + 1)
      p, s = corr_map[pair]
      idx1, idx2 = pair
      x1 = X1[:, idx1]
      x2 = X2[:, idx2]
      crow = i // ncol
      ccol = i % ncol
      if is_marker_pairs:
        color = 'salmon' if crow == ccol else 'blue'
      else:
        color = 'salmon' if ccol < top else 'blue'
      vs.plot_scatter(x=x1[ids],
                      y=x2[ids],
                      color=color,
                      ax=ax,
                      size=library[ids],
                      size_range=(6, 30),
                      legend_enable=False,
                      linewidths=0.,
                      cbar=False,
                      alpha=0.3)
      # additional title for first column
      ax.set_title(f"{om1_names[idx1]}\n$p={p:.2g}$ $s={s:.2g}$", fontsize=8)
      # beginning of every column
      if i % ncol == 0:
        ax.set_ylabel(f"{om2_names[idx2]}", fontsize=8, weight='bold')
    ## big title
    plt.suptitle(f"[x:{omic1.name}_y:{omic2.name}]{title}", fontsize=10)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
    ### store and return
    if return_figure:
      return fig
    self.add_figure(
        f"corr_{omic1.name}{'log' if log1 else 'raw'}_"
        f"{omic2.name}{'log' if log2 else 'raw'}", fig)
    return self

  #### Latents
  def plot_divergence(self,
                      X=OMIC.transcriptomic,
                      omic=OMIC.proteomic,
                      algo='tsne',
                      n_pairs=18,
                      ncol=6):
    r""" Select the most diverged pair within given `omic`, use `X` as
    coordinate and the pair's value as intensity for plotting the scatter
    heatmap. """
    om1 = OMIC.parse(X)
    om2 = OMIC.parse(omic)
    ## prepare the coordinate
    X = self.dimension_reduce(om1, n_components=2, algo=algo)
    n_points = X.shape[0]
    ## prepare the value
    y = self.numpy(om2)
    varnames = self.get_var_names(om2)
    ## check correlation type
    corr_fn = lambda m, n: (spearmanr(m, n, nan_policy='omit').correlation +
                            pearsonr(m, n)[0]) / 2
    ## create the correlation matrix
    corr_ids = []
    corr = []
    for i in range(y.shape[1]):
      for j in range(i + 1, y.shape[1]):
        corr_ids.append((i, j))
        corr.append(corr_fn(y[:, i], y[:, j]))
    ## sorting and select the smallest correlated pairs
    sort_ids = np.argsort(corr)[:int(n_pairs)]
    corr = np.array(corr)[sort_ids]
    corr_ids = np.array(corr_ids)[sort_ids]
    ## plotting
    nrow = int(np.ceil((n_pairs / ncol)))
    fig = plt.figure(figsize=(ncol * 3, nrow * 3))
    for idx, ((i, j), c) in enumerate(zip(corr_ids, corr)):
      name1 = varnames[i]
      name2 = varnames[j]
      y1 = y[:, i]
      y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
      y2 = y[:, j]
      y2 = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))
      val = y1 - y2
      vs.plot_scatter(X,
                      color='bwr',
                      size=20 if n_points < 1000 else (100000 / n_points),
                      val=val,
                      alpha=0.6,
                      cbar=True,
                      cbar_ticks=[name2, 'Others', name1],
                      cbar_horizontal=True,
                      fontsize=8,
                      ax=(nrow, ncol, idx + 1))
    ## adjust and save
    self.add_figure("divergence_%s_%s_%s" % (om1.name, om2.name, algo), fig)
    return self

  #### Basic statistic
  def plot_histogram(self,
                     omic=OMIC.proteomic,
                     bins=80,
                     log_norm=True,
                     var_names=None,
                     max_plots=100,
                     fig=None,
                     return_figure=False):
    r""" Plot histogram for each variable of given OMIC type """
    omic = OMIC.parse(omic)
    x = self.numpy(omic)
    bins = min(int(bins), x.shape[0] // 2)
    max_plots = int(max_plots)
    ### prepare the data
    var_ids = self.get_var_indices(omic)
    if var_names is None:
      var_names = var_ids.keys()
    var_names = np.array([i for i in var_names if i in var_ids])
    assert len(var_names) > 0, \
      f"No matching variables found for {omic.name}"
    # randomly select variables
    if len(var_names) > max_plots:
      rand = np.random.RandomState(seed=1)
      ids = rand.permutation(len(var_names))[:max_plots]
      var_names = var_names[ids]
    ids = [var_ids[i] for i in var_names]
    x = x[:, ids]
    ### the figures
    ncol = 8
    nrow = int(np.ceil(x.shape[1] / ncol))
    if fig is None:
      fig = vs.plot_figure(nrow=nrow * 2, ncol=ncol * 3, dpi=80)
    # plot
    for idx, (y, name) in enumerate(zip(x.T, var_names)):
      sparsity = sparsity_percentage(y, batch_size=2048)
      y = y[y != 0.]
      if log_norm:
        y = np.log1p(y)
      vs.plot_histogram(x=y,
                        bins=bins,
                        alpha=0.8,
                        ax=(nrow, ncol, idx + 1),
                        title=f"{name}\n({sparsity*100:.1f}% zeros)")
      fig.gca().tick_params(axis='y', labelleft=False)
    ### adjust and return
    fig.suptitle(f"{omic.name}")
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    if return_figure:
      return fig
    return self.add_figure(f"histogram_{omic.name}", fig)

  def plot_percentile_histogram(self,
                                omic=OMIC.transcriptomic,
                                n_hist=10,
                                title="",
                                outlier=0.001,
                                non_zeros=False,
                                fig=None):
    r""" Data is chopped into multiple percentile (`n_hist`) and the
    histogram is plotted for each percentile. """
    omic = OMIC.parse(omic)
    arr = self.numpy(omic)
    if non_zeros:
      arr = arr[arr != 0]
    n_percentiles = n_hist + 1
    n_col = 5
    n_row = int(np.ceil(n_hist / n_col))
    if fig is None:
      fig = vs.plot_figure(nrow=int(n_row * 1.5), ncol=20)
    self.assert_figure(fig)
    percentile = np.linspace(start=np.min(arr),
                             stop=np.max(arr),
                             num=n_percentiles)
    n_samples = len(arr)
    for i, (p_min, p_max) in enumerate(zip(percentile, percentile[1:])):
      min_mask = arr >= p_min
      max_mask = arr <= p_max
      mask = np.logical_and(min_mask, max_mask)
      a = arr[mask]
      _, bins = vs.plot_histogram(
          a,
          bins=120,
          ax=(n_row, n_col, i + 1),
          fontsize=8,
          color='red' if len(a) / n_samples < outlier else 'blue',
          title=f"{len(a)}(samples)  Range:[{p_min:.2g},{p_max:.2g}]")
      plt.gca().set_xticks(np.linspace(np.min(bins), np.max(bins), num=8))
    if len(title) > 0:
      plt.suptitle(title)
    plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
    self.add_figure(f'histogram{n_hist}_{omic.name}', fig)
    return self

  def plot_series(self,
                  omic1=OMIC.transcriptomic,
                  omic2=OMIC.proteomic,
                  var_names1='auto',
                  var_names2='auto',
                  log1=True,
                  log2=True,
                  fontsize=10,
                  title='',
                  return_figure=False):
    r""" Plot lines of 2 OMICs sorted in ascending order of `omic1` """
    import seaborn as sns
    ## prepare
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    omic1_ids = self.get_var_indices(omic1)
    omic2_ids = self.get_var_indices(omic2)
    if isinstance(var_names1, string_types) and var_names1 == 'auto':
      var_names1 = omic1.markers
    if isinstance(var_names2, string_types) and var_names2 == 'auto':
      var_names2 = omic2.markers
    ## filtering variables
    ids1 = []
    ids2 = []
    for v1, v2 in zip(var_names1, var_names2):
      i1 = omic1_ids.get(v1, None)
      i2 = omic2_ids.get(v2, None)
      if i1 is not None and i2 is not None:
        ids1.append(i1)
        ids2.append(i2)
    assert len(ids1) > 0, \
      (f"No variables found for omic1={omic1} var1={var_names1} "
       f"and omic2={omic2} var2={var_names2}")
    x1 = self.get_omic(omic1)[:, ids1]
    x2 = self.get_omic(omic2)[:, ids2]
    if log1:
      x1 = np.log1p(x1)
    if log2:
      x2 = np.log1p(x2)
    names1 = self.get_var_names(omic1)[ids1]
    names2 = self.get_var_names(omic2)[ids2]
    n_series = len(names1)
    ### prepare the plot
    colors = sns.color_palette(n_colors=2)
    fig = plt.figure(figsize=(12, n_series * 4))
    for idx in range(n_series):
      y1 = x1[:, idx]
      y2 = x2[:, idx]
      order = np.argsort(y1)
      ax = plt.subplot(n_series, 1, idx + 1)
      ## the second series
      ax.plot(y1[order],
              linewidth=1.8,
              color=colors[0],
              label=f"{omic1.name}-{names1[idx]}")
      ax.set_ylabel(f"{'log' if log1 else 'raw'}-{omic1.name}-{names1[idx]}",
                    color=colors[0])
      ax.set_xlabel(f"Cell in ascending order of {omic1.name}")
      ax.tick_params(axis='y', colors=colors[0], labelcolor=colors[0])
      ax.grid(False)
      ## the second series
      ax = ax.twinx()
      ax.plot(y2[order],
              linestyle='--',
              alpha=0.88,
              linewidth=1.2,
              color=colors[1])
      ax.set_ylabel(f"{'log' if log1 else 'raw'}-{omic2.name}-{names2[idx]}",
                    color=colors[1])
      ax.tick_params(axis='y', colors=colors[1], labelcolor=colors[1])
      ax.grid(False)
    ### finalize the figure style
    if len(title) > 0:
      plt.suptitle(title, fontsize=fontsize + 2)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0., 0.02, 1., 0.98])
    if return_figure:
      return fig
    return self.add_figure(f'series_{omic1.name}_{omic2.name}', fig)

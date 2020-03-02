from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from numbers import Number

import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from six import string_types

from odin import visual as vs
from odin.utils import as_tuple
from odin.visual import Visualizer, to_axis
from sisua.data.const import MARKER_GENES, OMIC
from sisua.data.utils import is_categorical_dtype


# ===========================================================================
# Helper
# ===========================================================================
def _process_omics(sco, omic, clustering=None):
  if isinstance(omic, OMIC):
    omic = omic.name
  else:
    try:
      omic = OMIC.parse(omic).name
    except ValueError:
      pass
  omic = str(omic)
  x = None
  if omic in sco.obs:
    x = sco.obs[omic].values
  elif omic in sco.obsm or omic == OMIC.transcriptomic.name:
    # Use Louvain community detection
    if isinstance(clustering, string_types):
      clustering = clustering.lower().strip()
      if 'louvain' in clustering:
        x = sco.louvain(omic)
        omic = omic + '_louvain'
      elif 'kmean' in clustering:
        x = sco.kmeans(omic)
        omic = omic + '_kmeans'
      else:
        raise ValueError(
            "No support for clustering algorithm '%s' of OMIC type '%s'" %
            (clustering, omic))
    # probabilistic embedding
    else:
      x = sco.numpy(omic)
      _, prob, _ = sco.probabilistic_embedding(omic)
      try:
        x = sco.labels(omic).values
        omic = sco.labels_name(omic)
      except KeyError:  # no variable name, just use raw integer values
        x = np.argmax(prob, axis=1)
  return omic, x


def _marker_genes(sco, marker_genes):
  if isinstance(marker_genes, Number):
    marker_genes = sco.top_genes(n_genes=int(marker_genes))
  elif marker_genes is None:
    marker_genes = sco.marker_genes
  else:
    marker_genes = as_tuple(marker_genes, t=string_types)
  assert all(g in sco.gene_id for g in marker_genes)
  return marker_genes


# ===========================================================================
# Main class
# ===========================================================================
class SingleCellVisualizer(Visualizer):

  @contextmanager
  def _swap_omic(self, omic):
    r""" Temporary change the main OMIC type to other than transcriptomic """
    if isinstance(omic, (OMIC, string_types)):
      omic = OMIC.parse(omic)
      x = self.numpy(omic)
    else:
      x = omic
    org_x = self._X
    self._X = x
    yield self
    self._X = org_x

  def plot_scatter(self,
                   X=OMIC.transcriptomic,
                   colorby=OMIC.proteomic,
                   markerby=OMIC.celltype,
                   clustering=None,
                   legend_loc='best',
                   algo='tsne',
                   ax=None):
    r""" Scatter plot of dimension using binarized protein labels

    Arguments:
      legend_loc: one of the following 'best', 'upper right', 'upper left',
        'lower left', 'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center'.
    """
    ax = vs.to_axis2D(ax, (8, 8))
    omic = OMIC.parse(X)
    omic_name = omic.name
    X = self.dimension_reduce(omic, n_components=2, algo=algo)
    _, colors = _process_omics(self, colorby, clustering=clustering)
    _, markers = _process_omics(self, markerby, clustering=clustering)
    if is_categorical_dtype(colors):  # categorical values
      fn = vs.plot_scatter
      kw = dict(color='b' if colors is None else colors)
    else:  # integral values
      fn = vs.plot_scatter_heatmap
      kw = dict(val=colors, colormap='bwr')
    fn(X,
       marker='.' if markers is None else markers,
       size=180000 / X.shape[0],
       alpha=0.8,
       legend_enable=True if legend_loc is not None else False,
       legend_loc=legend_loc,
       grid=False,
       ax=ax,
       title="[%s] %s-%s" % (algo, self.name.split('_')[0], omic_name),
       **kw)
    self.add_figure('scatter_%s_%s' % (omic_name, str(algo).lower()),
                    ax.get_figure())
    return self

  def plot_dotplot(self,
                   groupby=OMIC.transcriptomic,
                   marker_genes=None,
                   clustering=None,
                   rank_genes=0,
                   dendrogram=True,
                   standard_scale='var',
                   cmap='Reds',
                   log=False):
    marker_genes = _marker_genes(self, marker_genes)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    kw = dict(dendrogram=dendrogram,
              log=log,
              standard_scale=standard_scale,
              color_map=cmap)
    if rank_genes > 0:
      key = groupby + '_rank'
      if key not in self.uns:
        self.rank_genes_groups(groupby=groupby)
      axes = sc.pl.rank_genes_groups_dotplot(self,
                                             n_genes=int(rank_genes),
                                             key=key,
                                             **kw)
    else:
      axes = sc.pl.dotplot(self, var_names=marker_genes, groupby=groupby, **kw)
    self.add_figure('dotplot_%s' % ('log' if log else 'raw'), plt.gcf())
    return self

  def plot_stacked_violins(self,
                           groupby=OMIC.transcriptomic,
                           marker_genes=None,
                           clustering=None,
                           rank_genes=0,
                           dendrogram=False,
                           swap_axes=True,
                           standard_scale='var',
                           log=False):
    marker_genes = _marker_genes(self, marker_genes)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              log=log,
              standard_scale=standard_scale)
    if rank_genes > 0:
      key = groupby + '_rank'
      if key not in self.uns:
        self.rank_genes_groups(groupby=groupby)
      axes = sc.pl.rank_genes_groups_stacked_violin(self,
                                                    n_genes=int(rank_genes),
                                                    key=key,
                                                    **kw)
    else:
      axes = sc.pl.stacked_violin(self,
                                  var_names=marker_genes,
                                  groupby=groupby,
                                  **kw)
    self.add_figure('stacked_violin_%s' % ('log' if log else 'raw'), plt.gcf())
    return self

  def plot_heatmap(self,
                   X=OMIC.transcriptomic,
                   groupby=OMIC.transcriptomic,
                   marker_genes=None,
                   clustering=None,
                   rank_genes=0,
                   dendrogram=False,
                   swap_axes=False,
                   cmap=None,
                   standard_scale='var',
                   show_gene_labels=True,
                   log=True):
    r"""
    X : `OMIC` or `np.ndarray`. Input data for visualization
    """
    marker_genes = _marker_genes(self, marker_genes)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              standard_scale=standard_scale,
              show_gene_labels=bool(show_gene_labels),
              log=log)
    if cmap is not None:
      kw['cmap'] = cmap
    if rank_genes > 0:
      key = groupby + '_rank'
      if key not in self.uns:
        self.rank_genes_groups(groupby=groupby, n_genes=int(rank_genes))
      axes = sc.pl.rank_genes_groups_heatmap(self,
                                             n_genes=int(rank_genes),
                                             key=key,
                                             **kw)
    else:
      with self._swap_omic(X):
        axes = sc.pl.heatmap(self,
                             var_names=marker_genes,
                             groupby=groupby,
                             **kw)
    self.add_figure('heatmap_%s' % ('log' if log else 'raw'), plt.gcf())
    return self

  def plot_percentile_histogram(self,
                                n_hist=8,
                                omic=OMIC.transcriptomic,
                                title=None,
                                outlier=0.001,
                                non_zeros=False,
                                fig=None):
    r""" Data is chopped into multiple percentile (`n_hist`) and the
    histogram is plotted for each percentile. """
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
          title=("[%s]" % title if i == 0 else "") +
          "%d(samples)  Range:[%g, %g]" % (len(a), p_min, p_max))
      plt.gca().set_xticks(np.linspace(np.min(bins), np.max(bins), num=8))
    plt.tight_layout()
    self.add_figure('percentile_%dhistogram' % n_hist, fig)
    return self

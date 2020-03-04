from __future__ import absolute_import, division, print_function

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
  ## the omic provided already in observation
  if omic in sco.obs:
    x = sco.obs[omic].values
  ## processing of multi-dimensional OMIC for labeling and clustering
  elif omic in sco.omics:
    # Use Louvain community detection
    if isinstance(clustering, string_types):
      clustering = clustering.lower().strip()
      if 'louvain' in clustering:  # community detection
        x = sco.louvain(omic)
        omic = omic + '_louvain'
      else:  # clustering
        omic = sco.clustering(omic, algo=clustering, return_key=True)
        x = sco.obs[omic].values.get_values()
    # probabilistic embedding
    else:
      x = sco.numpy(omic)
      _, prob, _ = sco.probabilistic_embedding(omic)
      try:
        x = sco.labels(omic).values
        omic = sco.labels_name(omic)
      except KeyError:  # no variable name, just use raw integer values
        x = np.argmax(prob, axis=1)
  ## Exception
  else:
    raise ValueError("No support for omic: '%s' and clustering: '%s'" %
                     (omic, str(clustering)))
  return omic, x


def _process_varnames(sco, input_omic, var_names):
  input_omic = OMIC.parse(input_omic)
  with sco._swap_omic(input_omic) as sco:
    if isinstance(var_names, Number):
      var_names = sco.top_vars(n_vars=int(var_names))
    elif var_names is None:
      if input_omic == OMIC.transcriptomic:
        var_names = [i for i in sco.var_names if i in MARKER_GENES]
      else:  # just take all variables
        var_names = sco.var_names
    else:
      var_names = as_tuple(var_names, t=string_types)
    # check all var names are exist
    assert all(g in sco.gene_id for g in var_names)
  return input_omic, var_names


def _validate_arguments(kw):
  X = OMIC.parse(kw.get('X'))
  groupby = OMIC.parse(kw.get('groupby'))
  rank_genes = kw.get('rank_genes')
  clustering = kw.get('clustering')
  log = kw.get('log')
  if rank_genes:
    assert X == OMIC.transcriptomic, \
      "Only visualize transcriptomic in case of rank_genes>0, but given: %s" \
        % X.name
  title = '_'.join(i for i in [
      X.name, groupby.name,
      str(clustering), \
      ('rank' if rank_genes else ''),
      ('log' if log else 'raw')
  ] if len(i) > 0)
  return title


# ===========================================================================
# Main class
# ===========================================================================
class SingleCellVisualizer(sc.AnnData, Visualizer):

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

  def plot_stacked_violins(self,
                           X=OMIC.transcriptomic,
                           groupby=OMIC.transcriptomic,
                           var_names=None,
                           clustering='kmeans',
                           rank_genes=0,
                           dendrogram=False,
                           swap_axes=True,
                           standard_scale='var',
                           log=False):
    title = _validate_arguments(locals())
    X, var_names = _process_varnames(self, X, var_names)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              log=log,
              standard_scale=standard_scale)
    if rank_genes > 0:
      key = self.rank_vars_groups(groupby=groupby, n_vars=int(rank_genes))
      axes = sc.pl.rank_genes_groups_stacked_violin(self,
                                                    n_genes=int(rank_genes),
                                                    key=key,
                                                    **kw)
    else:
      with self._swap_omic(X):
        axes = sc.pl.stacked_violin(self,
                                    var_names=var_names,
                                    groupby=groupby,
                                    **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    plt.suptitle(title)
    self.add_figure('stacked_violin_%s' % title, fig)
    return self

  def plot_dotplot(self,
                   X=OMIC.transcriptomic,
                   groupby=OMIC.transcriptomic,
                   var_names=None,
                   clustering='kmeans',
                   rank_genes=0,
                   dendrogram=False,
                   standard_scale='var',
                   cmap='Reds',
                   log=True):
    title = _validate_arguments(locals())
    X, var_names = _process_varnames(self, X, var_names)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    kw = dict(dendrogram=dendrogram,
              log=log,
              standard_scale=standard_scale,
              color_map=cmap)
    if rank_genes > 0:
      key = self.rank_vars_groups(groupby=groupby, n_vars=int(rank_genes))
      axes = sc.pl.rank_genes_groups_dotplot(self,
                                             n_genes=int(rank_genes),
                                             key=key,
                                             **kw)
    else:
      with self._swap_omic(X):
        axes = sc.pl.dotplot(self, var_names=var_names, groupby=groupby, **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    plt.suptitle(title)
    self.add_figure('dotplot_%s' % title, fig)
    return self

  def plot_heatmap(self,
                   X=OMIC.transcriptomic,
                   groupby=OMIC.transcriptomic,
                   var_names=None,
                   clustering='kmeans',
                   rank_genes=0,
                   dendrogram=False,
                   swap_axes=False,
                   cmap='bwr',
                   standard_scale='var',
                   log=True):
    r"""
    X : `OMIC` or `np.ndarray`. Input data for visualization
    """
    title = _validate_arguments(locals())
    X, var_names = _process_varnames(self, X, var_names)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    ## select the right marker genes and omic
    kw = dict(dendrogram=dendrogram,
              swap_axes=bool(swap_axes),
              standard_scale=standard_scale,
              show_gene_labels=True if len(var_names) < 50 else False,
              log=log,
              cmap=cmap)
    ## plotting
    if rank_genes > 0:
      key = self.rank_vars_groups(groupby=groupby, n_vars=int(rank_genes))
      axes = sc.pl.rank_genes_groups_heatmap(self,
                                             n_genes=int(rank_genes),
                                             key=key,
                                             **kw)
    else:
      with self._swap_omic(X):
        axes = sc.pl.heatmap(self, var_names=var_names, groupby=groupby, **kw)
    ## reconfigure the axes
    fig = plt.gcf()
    plt.suptitle(title)
    self.add_figure('heatmap_%s' % title, fig)
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

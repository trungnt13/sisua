from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from six import string_types

from odin import search
from odin import visual as vs
from odin.bay.vi.utils import discretizing
from odin.utils import as_tuple
from odin.visual import Visualizer, to_axis
from sisua.data._single_cell_analysis import _OMICanalyzer
from sisua.data.const import (MARKER_ADT_GENE, MARKER_GENES, OMIC,
                              PROTEIN_PAIR_COMPARISON)
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
      kw['self'].name,\
      X.name,
      groupby.name,
      str(clustering),
      ('rank' if rank_genes else ''),
      ('log' if log else 'raw')
  ] if len(i) > 0)
  return title


def _check_proteomic(self):
  if not OMIC.proteomic in self.omics:
    raise ValueError(
        "Require proteomic for plotting the marker gene/protein, given: %s" %
        str(self.omics))


# ===========================================================================
# Main class
# ===========================================================================
class _OMICvisualizer(_OMICanalyzer, Visualizer):

  #### Scatter plot

  def plot_scatter(self,
                   X=OMIC.transcriptomic,
                   colorby=OMIC.proteomic,
                   markerby=None,
                   clustering='kmeans',
                   legend=True,
                   algo='tsne',
                   ax=None):
    r""" Scatter plot of dimension using binarized protein labels """
    ax = vs.to_axis2D(ax, (8, 8))
    omic = OMIC.parse(X)
    omic_name = omic.name
    ## prepare data
    X = self.dimension_reduce(omic, n_components=2, algo=algo)
    n_points = X.shape[0]
    _, colors = _process_omics(self,
                               colorby,
                               clustering=clustering,
                               allow_none=True)
    _, markers = _process_omics(self,
                                markerby,
                                clustering=clustering,
                                allow_none=True)
    ## ploting
    if is_categorical_dtype(colors):  # categorical values
      kw = dict(color='b' if colors is None else colors)
    else:  # integral values
      kw = dict(val=colors, color='bwr')
    vs.plot_scatter(\
      X,
      marker='.' if markers is None else markers,
      size=88 if n_points < 1000 else (120000 / n_points),
      alpha=0.8,
      legend_enable=bool(legend),
      legend_loc='best',
      grid=False,
      ax=ax,
      title="[%s] %s-%s" % (algo, self.name.split('_')[0], omic_name),
      **kw)
    self.add_figure('scatter_%s_%s' % (omic_name, str(algo).lower()),
                    ax.get_figure())
    return self

  #### Others plot

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

  #### Heatmap plot

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

  def plot_distance_heatmap(self,
                            X=OMIC.transcriptomic,
                            groupby=OMIC.transcriptomic,
                            var_names=None,
                            clustering='kmeans',
                            cmap='bwr',
                            legend=True,
                            log=True,
                            ax=None):
    r""" Heatmap of the distance among latents vector from different classes
    """
    title = _validate_arguments(locals())
    X, var_names = _process_varnames(self, X, var_names)
    groupby, _ = _process_omics(self, groupby, clustering=clustering)
    ax = vs.to_axis2D(ax)
    ## prepare the data
    vs.plot_distance_heatmap(self.numpy(X),
                             labels=self.numpy(groupby),
                             colormap=cmap,
                             legend_enable=legend,
                             lognorm=log,
                             ax=ax,
                             fontsize=8,
                             legend_ncol=3,
                             title=title)
    self.add_figure('distance_heatmap_%s' % title, ax.get_figure())
    return self

  #### Correlation plot

  def plot_correlation_matrix(self, n_genes=None):
    r""" Plot correlation matrix between mark gene-protein

    n_genes : an Integer or `None`. If `None`, automatically select all the
      marker genes, otherwise,
      select the top genes that correlates with most protein.
    """
    _check_proteomic(self)
    corr = self.omic_correlation(OMIC.proteomic)
    corr_map = {(x[0], x[1]): (x[2], x[3]) for x in corr}
    ##
    all_genes = {j: i for i, j in enumerate(self.var_names)}
    all_prots = {j: i for i, j in enumerate(self.omic_varnames(OMIC.proteomic))}
    n_prots = len(self.omic_var(OMIC.proteomic))
    ## finding the genes that most correlated to the other OMIC
    if n_genes is not None:
      n_genes = int(n_genes)
      top_genes = np.zeros(shape=(self.n_vars,), dtype=np.float32)
      for gene, prot, pearr, spearr in corr:
        top_genes[gene] += abs(pearr) + abs(spearr)
      ids = np.argsort(top_genes)[::-1][:n_genes]
    ## get the marker genes
    else:
      ids = []
      for gene in MARKER_GENES:
        if gene in all_genes:
          ids.append(all_genes[gene])
      n_genes = len(ids)
    assert n_genes > 0, "No gene for visualization"

    ## create correlation matrix
    pear = np.empty(shape=(n_genes, n_prots), dtype=np.float64)
    spear = np.empty(shape=(n_genes, n_prots), dtype=np.float64)
    for i, gene_idx in enumerate(ids):
      # this is redundant, but easy to read
      for j, om2_idx in enumerate(range(n_prots)):
        pair = (gene_idx, om2_idx)
        if pair in corr_map:
          pear[i, j], spear[i, j] = corr_map[pair]
        else:
          pear[i, j], spear[i, j] = 0, 0
    ## sorting the matrix
    sort_ids = search.diagonal_beam_search((pear + spear).T)
    pear = pear[sort_ids]
    spear = spear[sort_ids]
    # annotation
    gene_names = np.array([self.var_names[i] for i in ids])[sort_ids].tolist()
    prot_names = self.omic_varnames(OMIC.proteomic).tolist()
    ## helper for marking the marker gene-protein
    marker_position = []
    for prot, gene in MARKER_ADT_GENE.items():
      if prot in prot_names and gene in gene_names:
        marker_position.append((prot_names.index(prot), gene_names.index(gene)))

    def mark(ax):
      for x, y in marker_position:
        ax.text(x + 0.02,
                y + 0.04,
                s='O',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=60 / np.log1p(n_genes),
                color='green',
                alpha=0.5,
                weight='semibold')

    ## plotting
    styles = dict(cmap="bwr",
                  xticklabels=prot_names,
                  yticklabels=gene_names,
                  gridline=0.5,
                  fontsize=10 if n_genes < 100 else 8,
                  cbar=True)
    fig = plt.figure(figsize=(12, min(20, n_genes / 2)))
    mark(vs.plot_heatmap(pear, **styles, ax=(1, 2, 1), title="Pearson"))
    mark(vs.plot_heatmap(spear, **styles, ax=(1, 2, 2), title="Spearman"))

    ## adjusting
    fig.tight_layout()
    vs.plot_save()
    exit()
    ## store and return
    self.add_figure("correlation_matrix", fig)
    return self

  def plot_correlation(self,
                       log=False,
                       top=None,
                       bottom=None,
                       marker=True,
                       ncol=6):
    _check_proteomic(self)
    # get all correlations
    corr = self.omic_correlation(OMIC.proteomic)
    corr_map = {(x[0], x[1]): (x[2], x[3]) for x in corr}
    gene_names = self.var_names
    prot_names = self.omic_varnames(OMIC.proteomic)
    all_gene = {j: i for i, j in enumerate(gene_names)}
    all_prot = {j: i for i, j in enumerate(prot_names)}
    X1 = self.numpy(OMIC.transcriptomic)
    if log:
      X1 = np.log1p(X1)
    X2 = self.numpy(OMIC.proteomic)
    library = np.sum(X1, axis=1, keepdims=True)
    library = discretizing(library, n_bins=10, strategy='quantile').ravel()
    ### getting the marker pairs
    all_pairs = []
    is_marker = []
    if marker or (top is None and bottom is None):
      for prot, gene in sorted(MARKER_ADT_GENE.items()):
        if prot in all_prot and gene in all_gene:
          pair = (all_gene[gene], all_prot[prot])
          if pair in corr_map:
            all_pairs.append(pair)
            is_marker.append(True)
    # top pairs
    if top is not None:
      top = int(top)
      for x in corr[:top]:
        pair = (x[0], x[1])
        if pair not in all_pairs:
          all_pairs.append(pair)
          is_marker.append(False)
    # bottom pairs
    if bottom is not None:
      bottom = int(bottom)
      for x in corr[-bottom:]:
        pair = (x[0], x[1])
        if pair not in all_pairs:
          all_pairs.append(pair)
          is_marker.append(False)
    ### plotting
    n = len(all_pairs)
    ncol = int(ncol)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(ncol * 3, nrow * 3))
    for i, (pair, marker) in enumerate(zip(all_pairs, is_marker)):
      ax = plt.subplot(nrow, ncol, i + 1)
      p, s = corr_map[pair]
      gene = X1[:, pair[0]]
      prot = X2[:, pair[1]]
      vs.plot_scatter(x=prot,
                      y=gene,
                      color='salmon',
                      ax=ax,
                      size=library,
                      size_range=(8, 80),
                      legend_enable=False,
                      linewidths=0.,
                      cbar=False,
                      alpha=0.6)
      ax.set_title("%spearson:%.2g spearman:%.2g" %
                   ('*' if marker else '', p, s),
                   fontsize=10)
      ax.set_xlabel("Protein:" + prot_names[pair[1]], fontsize=10)
      ax.set_ylabel("Gene:" + gene_names[pair[0]], fontsize=10)
    fig.tight_layout()
    ### store and return
    self.add_figure("correlation_%s" % ('log' if log else 'raw'), fig)
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
    varnames = self.omic_varnames(om2)
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
  def plot_percentile_histogram(self,
                                omic=OMIC.transcriptomic,
                                n_hist=8,
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

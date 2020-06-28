from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import stats
from odin import visual as vs
from odin.utils import ArgController, catch_warnings_ignore
from sisua import OMIC, get_dataset, get_dataset_meta

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)


# ===========================================================================
# For data visualization
# ===========================================================================
def main(dsname, outpath):
  ds = get_dataset(dsname, override=False, verbose=True)
  # basic statistics
  for om in ds.omics:
    x = ds.numpy(om)
    var_names = ds.get_var_names(om)
    print(f"{om.name}:")
    print(f" shape: {x.shape}")
    print(f" n_var: {len(var_names)}")
    print(f" n_unique_var: {len(set(var_names))}")
    print(f" Sparsity: {stats.sparsity_percentage(x)}")
    print(f" n_zero_cols: {sum(np.sum(x, axis=0) == 0)}")
    print(f" n_zero_rows: {sum(np.sum(x, axis=1) == 0)}")
  # relevant factor or color OMIC
  for data_omic in (OMIC.transcriptomic, OMIC.chromatin):
    if data_omic in ds.omics:
      break
    else:
      data_omic = None
  for factor_omic in (OMIC.proteomic, OMIC.celltype, OMIC.disease,
                      OMIC.progenitor):
    if factor_omic in ds.omics:
      break
    else:
      factor_omic = None
  # plot using factor omic
  if data_omic is not None and factor_omic is not None:
    pairs = data_omic.marker_pairs(factor_omic)
    ds.plot_series(omic1=data_omic,
                   omic2=factor_omic,
                   var_names1=[i[0] for i in pairs],
                   var_names2=[i[1] for i in pairs],
                   log1=True,
                   log2=True)
    for clustering in ('louvain', 'kmeans', 'knn'):
      ds.plot_heatmap(X=data_omic, group_by=factor_omic, clustering=clustering)
      ds.plot_stacked_violins(X=data_omic,
                              group_by=factor_omic,
                              clustering=clustering)
  # plotting
  for om in ds.omics:
    X = ds.numpy(om)
    var_names = ds.get_var_names(om)
    markers = [i for i in om.markers if i in var_names]
    print(f"Plotting OMIC: '{om.name}' ...")
    print(f" Markers: {', '.join(markers)}")
    ds.plot_histogram(om)
    ds.plot_percentile_histogram(om, n_hist=20, title=f"{om.name}")
    for clustering in ('louvain', 'kmeans', 'knn'):
      for algo in ('pca', 'tsne', 'umap'):
        print(f" plot scatter: {clustering} - {algo}")
        ds.plot_scatter(X=om,
                        color_by=factor_omic,
                        clustering=clustering,
                        fig=(8, 8),
                        dimension_reduction=algo)
    if X.shape[1] < 100:
      ds.plot_spearman_matrix(omic1=om,
                              omic2=om,
                              var_names1=markers,
                              var_names2=markers,
                              is_marker_pairs=True)
      ds.plot_heatmap(X=om, group_by=None)
    if factor_omic is not None and factor_omic != om:
      ds.plot_spearman_matrix(omic1=om, omic2=factor_omic)
      ds.plot_pearson_matrix(omic1=om, omic2=factor_omic)
      ds.plot_mutual_information(omic1=om, omic2=factor_omic)
  # save the figures
  if os.path.isfile(outpath):
    raise ValueError(f"Output path at '{outpath}' must be a directory.")
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  ds.save_figures(path=outpath, dpi=100, separate_files=True, verbose=True)


# ===========================================================================
# Argument parsing
# ===========================================================================
if __name__ == "__main__":
  all_dataset = list(get_dataset_meta().keys())
  args = ArgController(print_parsed=True\
    ).add("dsname", f"all available datasets: {', '.join(all_dataset)}"\
    ).add("-path", "Output directory", '/tmp'\
    ).parse()
  dsname = args.dsname
  path = args.path
  assert dsname in all_dataset, \
    (f"No support for dataset with name: {dsname}, "
     f"all available datasets are: {all_dataset}")
  path = os.path.join(path, dsname)
  if not os.path.exists(path):
    os.makedirs(path)
  main(dsname, outpath=path)

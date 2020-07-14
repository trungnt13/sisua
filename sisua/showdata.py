from __future__ import absolute_import, division, print_function

import os
import shutil
from multiprocessing import Process

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import stats
from odin import visual as vs
from odin.utils import ArgController, catch_warnings_ignore
from sisua import OMIC, SingleCellOMIC, get_dataset, get_dataset_meta

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# For data visualization
# ===========================================================================
EXCLUDE_OMICS = [OMIC.pmhc, OMIC.celltype]
# maximum number of cells for visualization
N_CELLS = 12000


def save(ds: SingleCellOMIC, outpath):
  if not os.path.isdir(outpath):
    raise ValueError(f"Output path at '{outpath}' must be a directory.")
  ds.save_figures(path=outpath, dpi=100, separate_files=True, verbose=True)


def save_fig(fig, outpath):
  fig.savefig(outpath, dpi=100)
  plt.close(fig)
  print(f"Saved figure: {outpath}")


def main(dsname, outpath):
  ds = get_dataset(dsname, override=False, verbose=True)
  ds, _ = ds.split(train_percent=N_CELLS / ds.shape[0])
  print("*Save path:", outpath)
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
    if pairs is not None:
      kw = dict(omic1=data_omic,
                omic2=factor_omic,
                var_names1=[i[0] for i in pairs],
                var_names2=[i[1] for i in pairs])
      figname = f"{data_omic.name}_{factor_omic.name}.png"
      ds.plot_series(log1=True, log2=True, **kw)
      save_fig(ds.plot_spearman_matrix(**kw, return_figure=True),
               os.path.join(outpath, f"marker_spearman_{figname}"))
      save_fig(ds.plot_pearson_matrix(**kw, return_figure=True),
               os.path.join(outpath, f"marker_pearson_{figname}"))
      save_fig(ds.plot_mutual_information(**kw, return_figure=True),
               os.path.join(outpath, f"marker_mi_{figname}"))
    kw = dict(X=data_omic,
              group_by=factor_omic,
              var_names='auto',
              clustering='kmeans')
    ds.plot_heatmap(**kw)
    ds.plot_stacked_violins(**kw)
    save(ds, outpath)
  ### plotting for each OMIC type
  for om in ds.omics:
    if om in EXCLUDE_OMICS:
      continue
    X = ds.numpy(om)
    var_names = ds.get_var_names(om)
    markers = om.markers
    if markers is not None:
      markers = [i for i in markers if i in var_names]
    print(f"Plotting OMIC: '{om.name}' ...")
    print(f" Markers: {markers}")
    ds.plot_histogram(om)
    ds.plot_percentile_histogram(om, n_hist=20, title=f"{om.name}")
    # scatter plot
    if factor_omic is None or om != factor_omic:
      for clustering in ('kmeans',):
        for algo in ('tsne', 'umap'):
          print(f" plot scatter: {om}-{factor_omic} {clustering}-{algo}")
          ds.plot_scatter(X=om,
                          color_by=factor_omic,
                          clustering=clustering,
                          fig=(8, 8),
                          dimension_reduction=algo)
      save(ds, outpath)
    # scatter plot with celltype
    for cat_om in (OMIC.celltype, OMIC.disease, OMIC.progenitor):
      if cat_om in ds.omics and cat_om != factor_omic:
        for algo in ('tsne', 'umap'):
          print(f" plot scatter: {om}-{cat_om} {algo}")
          ds.plot_scatter(X=om,
                          color_by=cat_om,
                          fig=(8, 8),
                          clustering='kmeans',
                          dimension_reduction=algo)
          ds.plot_heatmap(X=om, group_by=cat_om)
    # correlation within the same omic
    if X.shape[1] < 100:
      kw = dict(omic1=om,
                omic2=om,
                var_names1=om.markers,
                var_names2=om.markers,
                is_marker_pairs=True)
      ds.plot_spearman_matrix(**kw)
      ds.plot_pearson_matrix(**kw)
      ds.plot_heatmap(X=om, group_by=None)
    # correlation with factor_omic
    if factor_omic is not None and factor_omic != om:
      kw = dict(omic1=om,
                omic2=factor_omic,
                var_names1=om.markers,
                var_names2=factor_omic.markers,
                is_marker_pairs=False)
      ds.plot_spearman_matrix(**kw)
      ds.plot_pearson_matrix(**kw)
      ds.plot_mutual_information(**kw)
    save(ds, outpath)


# ===========================================================================
# Argument parsing
# ===========================================================================
def call_main(dsname, outpath):
  try:
    main(dsname=dsname, outpath=outpath)
  except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"error:'{e}'\nds:'{dsname}'\npath:'{outpath}'")


if __name__ == "__main__":
  all_dataset = list(get_dataset_meta().keys())
  args = ArgController(print_parsed=True\
    ).add("dsname", f"all available datasets: {', '.join(all_dataset)}"\
    ).add("-path", "Output directory", '/tmp'\
    ).parse()
  all_dsname = args.dsname.split(',')
  path = args.path

  for dsname in all_dsname:
    if dsname not in all_dataset:
      print(f"No support for dataset with name: {dsname}, "
            f"all available datasets are: {all_dataset}")
      continue
    outpath = os.path.join(path, dsname)
    # override exists path
    if os.path.exists(outpath):
      shutil.rmtree(outpath)
    if not os.path.exists(outpath):
      os.makedirs(outpath)
    # run with multiprocessing
    p = Process(target=call_main, args=(dsname, outpath))
    p.start()
    p.join()

# example:
# python showdata.py 8kx,callx,mpalx,eccx,vdj4x

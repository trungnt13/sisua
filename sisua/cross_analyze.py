from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle
from six import string_types
from collections import defaultdict
from itertools import product

import seaborn as sns
from matplotlib import pyplot as plt
from sisua.data import get_dataset, get_dataset_summary

from odin import visual as vs
from odin.utils import as_tuple, ctext
from odin.ml import fast_tsne, fast_pca

import numpy as np

from sisua.data import get_dataset
from sisua.data.path import EXP_DIR
from sisua.data.utils import standardize_protein_name
from sisua.analysis import Posterior

def cross_analyze(datasets, outpath, model='movae', verbose=False):
  datasets = as_tuple(datasets, t=string_types)
  assert len(datasets) > 1, \
  "Require more than one datasets for cross analysis"
  if not os.path.exists(outpath):
    os.mkdir(outpath)
  # ====== load datasets ====== #
  all_datasets = {name: get_dataset(name)[0]
                  for name in datasets}
  all_datasets = [
  (name, dict(X=ds['X'][:],
              X_col=ds['X_col'],
              X_row=ds['X_row'],
              y=ds['y'],
              y_col=np.array([standardize_protein_name(i)
                              for i in ds['y_col']]),
  ))
  for name, ds in all_datasets.items()]
  # ====== check gene expression is matching ====== #
  genes = all_datasets[0][1]['X_col']
  for name, ds in all_datasets:
    assert np.all(ds['X_col'] == genes), "Set of training genes mis-match"
  # ====== get the list of all overlapping protein ====== #
  all_proteins = set(all_datasets[0][1]['y_col'])
  for name, ds in all_datasets:
    all_proteins &= set(ds['y_col'])
  # ====== only select certain protein ====== #
  if verbose:
    print("Datasets       :", ctext(', '.join(datasets), 'yellow'))
    print("Model          :", ctext(model, 'yellow'))
    print("Shared proteins:", ctext(', '.join(all_proteins), 'yellow'))
    for name, ds in all_datasets:
      print(" ", ctext(name, 'cyan'))
      print("   X    :", ds['X'].shape)
      print("   X_col:", ds['X_col'])
      print("   y    :", ds['y'].shape)
      print("   y_col:", ', '.join(ds['y_col']))
  # ====== load all the model ====== #
  all_models = []
  for ds_name in datasets:
    if verbose:
      print("Search model for dataset '%s' ..." % ctext(ds_name, 'yellow'))
    exp_path = os.path.join(EXP_DIR, ds_name)
    for model_name in os.listdir(exp_path):
      if model_name.split('_')[0] == model:
        path = os.path.join(exp_path, model_name, 'model.pkl')
        if os.path.exists(path):
          with open(path, 'rb') as f:
            m = pickle.load(f)
            all_models.append(m)
            if verbose:
              print(" ", ctext(m.id, 'cyan'))
  if verbose:
    print("%s datasets and %s models => %s experiments" % (
        ctext(len(all_datasets), 'yellow'),
        ctext(len(all_models), 'yellow'),
        ctext(len(all_datasets) * len(all_models), 'yellow'),
    ))
  # ====== start generate analysis ====== #
  for ds_name, ds in all_datasets:
    y_true = {i: j
              for i, j in zip(ds['y_col'], ds['y'].T)
              if i in all_proteins}
    # preserve the order in all_proteins
    y_true = np.hstack([y_true[i][:, np.newaxis] for i in all_proteins])

    # path for the dataset
    ds_path = os.path.join(outpath, ds_name)
    if not os.path.exists(ds_path):
      os.mkdir(ds_path)

    for infer in all_models:
      ds_infer = infer.configs['dataset']
      path = os.path.join(ds_path, '%s_%s.pdf' %
        (ds_infer.replace('_', ''), infer.short_id))
      # analysis
      pos = Posterior(infer, ds=ds)
      pos.new_figure(
      ).plot_latents_scatter(size=4
      ).plot_latents_heatmap(
      ).plot_correlation_series(
      )
      # protein series
      if infer.is_semi_supervised:
        y_pred = {i: j
                  for i, j in zip(dict(all_datasets)[ds_infer]['y_col'],
                                  infer.predict_y(ds['X']).T)
                  if i in all_proteins}
        y_pred = np.hstack([y_pred[i][:, np.newaxis] for i in all_proteins])
        pos.plot_protein_series(
            y_true_new=y_true, y_pred_new=y_pred, labels_new=all_proteins)
      # save plot and show log
      pos.save_plots(path, dpi=80)
      if verbose:
        print("Data:%s - Model:%s" % (
            ctext(ds_name, 'yellow'),
            ctext(ds_infer, 'yellow')))
        print(" Outpath:", ctext(path, 'cyan'))

cross_analyze(datasets=['cross8k_ly', 'crossecc_ly'],
              outpath='/tmp/cross',
              model='movae',
              verbose=True)

from __future__ import print_function, division, absolute_import

import os
import re
import time
import pickle
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

import numpy as np
import pandas as pd

from odin.utils import (ArgController, ctext)
from odin.utils.cache_utils import cache_memory

from sisua import set_verbose
from sisua.data import EXP_DIR, UNIVERSAL_RANDOM_SEED
from sisua.utils import filtering_experiment_path

from sisua.analysis.latent_benchmarks import (
    streamline_classifier, clustering_scores)
from sisua.analysis.imputation_benchmarks import (
    imputation_score, imputation_mean_score, imputation_std_score,
    ProteinGeneAnalysis)

# ===========================================================================
# Helper
# ===========================================================================
@cache_memory
def get_model(path):
  try:
    with open(os.path.join(path, 'model.pkl'), 'rb') as f:
      return pickle.load(f)
  except Exception:
    return None

@cache_memory
def name_2_attr(path):
  attr = {}
  name = os.path.basename(path).split('_')
  model = get_model(path)
  config = model.config

  attr['model'] = name[0]

  attr['xnorm'] = config['xnorm']
  attr['xdist'] = config['xdist']
  attr['xdrop'] = config['xdrop']

  attr['ynorm'] = config['ynorm']
  attr['ydist'] = config['ydist']

  attr['tnorm'] = config['tnorm']

  attr['supervised'] = (0 if name[0] == 'vae' else
                        model.supervised_percent)
  return attr

def normalize_score_name(name):
  name = name.split(':')[0]
  name = name.split('/')[-1]
  if '_loss' == name[-5:]:
    name = 'loss'
  if name == "NLLK":
    name = "NLLK_X"
  elif name == 'KLqp':
    name = "KLqp_X"
  elif name == 'reconstruction_mse':
    name = 'loss'
  elif 'Multitask_Loss' in name:
    name = 'loss'
  elif 'ReconstructionLoss' in name:
    name = "NLLK_X"
  elif 'SupervisedLoss' in name:
    name = "NLLK_Y"
  return name

def dfs_to_excel(*dfs, outpath, sheet_name='Sheet1'):
  # , engine='xlsxwriter'
  writer = pd.ExcelWriter(outpath)
  spaces = 2
  row = 0
  for i, df in enumerate(dfs):
    assert isinstance(df, pd.DataFrame)
    df.to_excel(writer, sheet_name=sheet_name,
                header=True,
                float_format='%.2f', index=False,
                startrow=row, startcol=0)
    row = row + len(df.index) + spaces
  writer.save()

# ===========================================================================
# Perform analysis
# ===========================================================================
def main():
  N_MCMC_SAMPLES = 100
  # ====== read data ====== #
  args = ArgController(
  ).add('ds', 'name of the dataset: mnist, pbmc_citeseq, pbmc_10xPP, cbmc, facs_2, facs_5, facs_7'
  ).add('-incl', 'keywords for including', ''
  ).add('-excl', 'keywords for excluding', ''
  ).parse()

  set_verbose(False)

  global ds_name, gene_ds, prot_ds
  all_exp, ds_name, gene_ds, prot_ds = filtering_experiment_path(
      ds_name_or_path=args.ds,
      incl_keywords=args.incl, excl_keywords=args.excl,
      return_dataset=True,
      print_log=True)

  X_train_org = gene_ds.get_data('train', dropout=0)
  X_test_org = gene_ds.get_data('test', dropout=0)
  y_train, y_test = prot_ds.get_data('train'), prot_ds.get_data('test')
  protein_markers = prot_ds.col_name
  n_protein = len(protein_markers)

  # ====== output path ====== #
  results_path = os.path.join(EXP_DIR, 'results')
  if not os.path.exists(results_path):
    os.mkdir(results_path)
  output_path = os.path.join(results_path, ds_name)
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  print("Saving output analysis at:", ctext(output_path, 'lightcyan'))
  # ====== start evaluating ====== #
  train_dataframe = defaultdict(list)
  test_dataframe = defaultdict(list)
  correlation_train = defaultdict(list)
  correlation_test = defaultdict(list)

  for corruption, all_path in all_exp.items():
    print("Processing Corruption: %s - %s experiments" %
      (ctext(corruption, 'lightyellow'), ctext(len(all_path), 'yellow')))

    for path in all_path:
      start_time = time.time()
      row_train = dict()
      row_test = dict()
      # load the model
      model = get_model(path)
      if model is None: # just ignore error model
        print(" Ignore '%s'!" % path)
        continue
      # convert to specific name
      attr = name_2_attr(path)
      row_train.update(attr)
      row_test.update(attr)
      # protein-genes mapping analysis
      res = ProteinGeneAnalysis.get_marker_genes_analysis(infer=model)
      if res is not None:
        corr_train = dict(row_train)
        corr_test = dict(row_test)
        trn, tst = res.get_spearman_correlation()
        for prot_gene, org, crp, imp in trn:
          corr_train[prot_gene] = imp
        for prot_gene, org, crp, imp in tst:
          corr_test[prot_gene] = imp
        correlation_train[corruption].append(corr_train)
        correlation_test[corruption].append(corr_test)
      # imputation test
      corrupted_rate = model.config.get('ximpu', 0)
      corrupted_dist = model.config.get('cdist', 'uniform')
      X_train = gene_ds.get_data('train',
                                 dropout=corrupted_rate,
                                 distribution=corrupted_dist)
      X_test = gene_ds.get_data('test',
                                dropout=corrupted_rate,
                                distribution=corrupted_dist)
      # log-likelihood
      for name, val in model.score(X_train, y_train,
                                   n_mcmc_samples=N_MCMC_SAMPLES).items():
        row_train['.' + normalize_score_name(name)] = val
      for name, val in model.score(X_test, y_test,
                                   n_mcmc_samples=N_MCMC_SAMPLES).items():
        row_test['.' + normalize_score_name(name)] = val
      # latent space
      Z_train = model.predict_Z(X_train)
      Z_test = model.predict_Z(X_test)
      for name, val in clustering_scores(
          latent=Z_train, labels=y_train, n_labels=n_protein).items():
        row_train['_' + name] = val
      for name, val in clustering_scores(
          latent=Z_test, labels=y_test, n_labels=n_protein).items():
        row_test['_' + name] = val
      # imputation
      V_train = model.predict_V(X_train, n_mcmc_samples=N_MCMC_SAMPLES)
      V_test = model.predict_V(X_test, n_mcmc_samples=N_MCMC_SAMPLES)
      if V_train is not None:
        V_train = V_train[0]
        V_test = V_test[0]
      else:
        V_train = model.predict_W(X_train, n_mcmc_samples=N_MCMC_SAMPLES)[0]
        V_test = model.predict_W(X_test, n_mcmc_samples=N_MCMC_SAMPLES)[0]
      row_train['d'] = imputation_score(
          original=X_train_org, imputed=V_train)
      row_train['d_mean'] = imputation_mean_score(
          original=X_train_org, corrupted=X_train, imputed=V_train)
      row_train['d_std'] = imputation_std_score(
          original=X_train_org, corrupted=X_train, imputed=V_train)
      row_test['d'] = imputation_score(
          original=X_test_org, imputed=V_test)
      row_test['d_mean'] = imputation_mean_score(
          original=X_test_org, corrupted=X_test, imputed=V_test)
      row_test['d_std'] = imputation_std_score(
          original=X_test_org, corrupted=X_test, imputed=V_test)
      # latent classifier
      train, test = streamline_classifier(Z_train, y_train, Z_test, y_test,
                                          train_results=True,
                                          labels_name=protein_markers,
                                          show_plot=False)
      row_train.update(train)
      row_test.update(test)
      print(" Finish '%s' in %.2f(s)" %
        (ctext(os.path.basename(path), 'yellow'), time.time() - start_time))
      # store
      train_dataframe[corruption].append(row_train)
      test_dataframe[corruption].append(row_test)

  # ====== save all scores ====== #
  fn_sort = lambda x: (x['tnorm'] +
                       x['xnorm'] +
                       ('1' if 'vae' in x['model'] else '2') +
                       x['model'])
  dsname = args.ds.lower()

  for corrupt_name in sorted(train_dataframe.keys()):
    # for scores
    train_df = sorted(train_dataframe[corrupt_name], key=fn_sort)
    test_df = sorted(test_dataframe[corrupt_name], key=fn_sort)
    dfs_to_excel(pd.DataFrame(train_df), pd.DataFrame(test_df),
                 outpath=os.path.join(output_path, '%s_%s.xlsx' % (corrupt_name, dsname)),
                 sheet_name="Results")
    # for correlation
    train_df = sorted(correlation_train[corrupt_name], key=fn_sort)
    test_df = sorted(correlation_test[corrupt_name], key=fn_sort)
    dfs_to_excel(pd.DataFrame(train_df), pd.DataFrame(test_df),
                 outpath=os.path.join(output_path, '%s_%s_corr.xlsx' % (corrupt_name, dsname)),
                 sheet_name="Results")

  print("Saving score files to:", ctext(output_path, 'cyan'))

if __name__ == '__main__':
  main()

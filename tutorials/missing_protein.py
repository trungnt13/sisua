from __future__ import print_function, division, absolute_import

import os

import numpy as np

from sisua.data import get_dataset
from sisua.data.utils import standardize_protein_name

# ===========================================================================
# Configurations
# ===========================================================================
FIGURE_PATH = '/tmp/missing_protein'

corruption_rate = 0.25
corruption_dist = 'binomial'

n_epoch = 200
batch_size = 128

if not os.path.exists(FIGURE_PATH):
  os.mkdir(FIGURE_PATH)

# ===========================================================================
# Load dataset
# ===========================================================================
# for evaluating
ds_eval, gene_eval, prot_eval = get_dataset('cross8k_ly')
# for evaluating cross-dataset
ds_cross, gene_cross, prot_cross = get_dataset('crossecc_ly')

n_genes = gene_eval.feat_dim
eval_ds = dict(
    X_train=gene_eval.X_train,
    X_test=gene_eval.X_test,
    X_col=gene_eval.col_name,
    y_train=prot_eval.X_train,
    y_test=prot_eval.X_test,
    y_col=prot_eval.col_name)

cross_ds = dict(
    X_train=gene_cross.X_train,
    X_test=gene_cross.X_test,
    X_col=gene_cross.col_name,
    y_train=prot_cross.X_train,
    y_test=prot_cross.X_test,
    y_col=prot_cross.col_name)

# ===========================================================================
# Helper
# ===========================================================================
def train_and_evaluate(ds_name, exp_name):
  from sisua.inference import InferenceSCVAE, InferenceSCVI, InferenceSISUA
  from sisua.analysis import Posterior, ResultsSheet

  ds, gene, prot = get_dataset(ds_name)

  # make sure gene expression stay the same
  assert np.all(gene.X_train == gene_eval.X_train) and \
  np.all(gene.X_test == gene_eval.X_test)

  print("\n======== Running experiment ========")
  print("Training %d-proteins:" % len(prot.col_name),
    ', '.join([standardize_protein_name(i) for i in prot.col_name]))
  print("Testing  %d-proteins:" % len(prot_eval.col_name),
    ', '.join([standardize_protein_name(i) for i in prot_eval.col_name]))

  n_prots = prot.feat_dim

  # ====== Main model training ====== #
  models = [
      InferenceSCVAE(gene_dim=n_genes),
      InferenceSCVI(gene_dim=n_genes),
      InferenceSISUA(gene_dim=n_genes, prot_dim=n_prots),
  ]
  for m in models:
    m.fit(X=gene.X_train,
          y=prot.X_train if m.is_semi_supervised else None,
          corruption_rate=corruption_rate, corruption_dist=corruption_dist,
          n_epoch=n_epoch, batch_size=batch_size,
          detail_logging=False)

  # ====== evaluation ====== #
  pos = [Posterior(m, ds=eval_ds)
         for m in models]

  res = ResultsSheet(pos, verbose=True)

  res.plot_learning_curves(
  ).save_plots(
      os.path.join(FIGURE_PATH, 'learning_curves_%s.pdf' % exp_name))

  res.plot_correlation_series(
  ).save_plots(
      os.path.join(FIGURE_PATH, 'correlation8k_%s.pdf' % exp_name))

  res.plot_latents_scatter(
  ).save_plots(
      os.path.join(FIGURE_PATH, 'latent8k_%s.pdf' % exp_name))

  res.plot_scores(score_type='classifier'
  ).save_plots(
      os.path.join(FIGURE_PATH, 'classifier8k_%s.pdf' % exp_name))

  # ====== cross ds ====== #
  pos = [Posterior(m, ds=cross_ds)
         for m in models]

  res = ResultsSheet(pos, verbose=True)

  res.plot_correlation_series(
  ).save_plots(
      os.path.join(FIGURE_PATH, 'correlationECC_%s.pdf' % exp_name))

  res.plot_latents_scatter(
  ).save_plots(
      os.path.join(FIGURE_PATH, 'latentECC_%s.pdf' % exp_name))

  res.plot_scores(score_type='classifier'
  ).save_plots(
      os.path.join(FIGURE_PATH, 'classifierECC_%s.pdf' % exp_name))

# ===========================================================================
# Main experiments
# Run multiple experiments at once using multiprocessing
# ===========================================================================
from multiprocessing import Pool

jobs = [('cross8k_' + i, i)
        for i in ('onlycd8', 'ly', 'nocd48', 'nocd4', 'nocd8')]
p = Pool(processes=3)
p.starmap(train_and_evaluate, jobs)
p.join()
p.close()

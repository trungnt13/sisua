from __future__ import print_function, division, absolute_import

import os
from itertools import product

import numpy as np
from odin.utils import ctext, catch_warnings_ignore

from sisua.data import get_dataset, SingleCellOMIC
from sisua.data.utils import standardize_protein_name

# ===========================================================================
# Configurations
# ===========================================================================
FIGURE_PATH = '/tmp/cross_datasets'

corruption_rate = 0.25
corruption_dist = 'binomial'

n_epoch = 1
batch_size = 128

if not os.path.exists(FIGURE_PATH):
  os.mkdir(FIGURE_PATH)

# ===========================================================================
# Load dataset
# ===========================================================================
all_datasets = {
    '8k': get_dataset('cross8k_ly'),
    'ecc': get_dataset('crossecc_ly')
}

# ====== check gene expression is matching ====== #
genes_name = None
all_proteins = None

for name, (ds, gene, prot) in all_datasets.items():
  if genes_name is None:
    genes_name = gene.col_name
  else:
    assert np.all(gene.col_name == genes_name), "Set of training genes mis-match"

  prots_name = set([standardize_protein_name(i)
                    for i in prot.col_name])
  if all_proteins is None:
    all_proteins = prots_name
  else:
    all_proteins &= prots_name

all_proteins = sorted(all_proteins)
print("Shared proteins:", ctext(', '.join(all_proteins), 'yellow'))

n_genes = len(genes_name)
# ===========================================================================
# Train all the models
# ===========================================================================
def train_and_evaluate(model_name, train_ds):
  if model_name == 'dca':
    from sisua.inference import InferenceDCA as Inference
  elif model_name == 'scvae':
    from sisua.inference import InferenceSCVAE as Inference
  elif model_name == 'sisua':
    from sisua.inference import InferenceSISUA as Inference
  elif model_name == 'scvi':
    from sisua.inference import InferenceSCVI as Inference
  else:
    raise NotImplementedError
  from sisua.analysis import Posterior

  outpath = os.path.join(FIGURE_PATH, '%s_train%s' %
                         (model_name, train_ds.upper()))
  if not os.path.exists(outpath):
    os.mkdir(outpath)

  print("\n======== Running experiment ========")
  print("Model     :", ctext(model_name, 'cyan'))
  print("Inference :", ctext(Inference, 'cyan'))
  print("Train data:", ctext(train_ds, 'cyan'))
  print("Out path  :", ctext(outpath, 'cyan'))

  ds, gene, prot = all_datasets[train_ds]
  n_prots = prot.feat_dim
  org_prot = [standardize_protein_name(i) for i in prot.col_name]

  # ====== Main model training ====== #
  if model_name == 'sisua':
    model = Inference(gene_dim=n_genes, prot_dim=n_prots)
  else:
    model = Inference(gene_dim=n_genes)
  model.fit(X=gene.X_train,
            y=prot.X_train if model.is_semi_supervised else None,
            corruption_rate=corruption_rate, corruption_dist=corruption_dist,
            n_epoch=n_epoch, batch_size=batch_size,
            detail_logging=False)

  # ====== start evaluation ====== #
  for name, (ds, gene, prot) in all_datasets.items():
    y_true = {
        i: j
        for i, j in zip([standardize_protein_name(i) for i in prot.col_name],
                        ds['y'].T)
        if i in all_proteins}
    # preserve the same order of all_proteins
    y_true = np.hstack([y_true[i][:, np.newaxis]
                        for i in all_proteins])
    prot = SingleCellOMIC(matrix=y_true,
                           rowname=ds['X_row'], colname=all_proteins)

    # create a mixed Posterior
    pos = Posterior(model, ds=dict(X_train=gene.X_train,
                                   X_test=gene.X_test,
                                   X_col=gene.col_name,
                                   y_train=prot.X_train,
                                   y_test=prot.X_test,
                                   y_col=prot.col_name))
    # a lot of figures so RuntimeWarning about maximum amount
    # of figure will be appeared
    with catch_warnings_ignore(RuntimeWarning):
      # analysis
      pos.new_figure(
      ).plot_latents_binary_scatter(size=4
      ).plot_latents_distance_heatmap(
      ).plot_correlation_marker_pairs()
      # protein series
      if model.is_semi_supervised:
        y_true = pos.y_test
        y_pred = model.predict_y(pos.X_test)
        y_pred = {i: j
                  for i, j in zip(org_prot, y_pred.T)
                  if i in all_proteins}
        y_pred = np.hstack([y_pred[i][:, np.newaxis]
                            for i in all_proteins])
        pos.plot_protein_predicted_series(
            y_true_new=y_true, y_pred_new=y_pred, labels_new=all_proteins)
        for prot_name in all_proteins:
          pos.plot_protein_scatter(protein_name=prot_name,
            y_true_new=y_true, y_pred_new=y_pred, labels_new=all_proteins)
      # save plot and show log
      pos.save_plots(os.path.join(outpath, '%s.pdf' % name), dpi=80)

# ===========================================================================
# Main experiments
# Run multiple experiments at once using multiprocessing
# ===========================================================================
from multiprocessing import Pool

jobs = [(i, j)
        for i, j in product(['dca', 'scvi', 'scvae', 'sisua'],
                            ['8k', 'ecc'])]
p = Pool(processes=3)
p.starmap(train_and_evaluate, jobs)
p.close()

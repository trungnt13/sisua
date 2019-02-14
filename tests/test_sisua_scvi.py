# dataset.expression.index
import os
os.environ['ODIN'] = 'gpu,float32'
import time
from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from odin.utils import one_hot, batching, ctext
from odin.ml import fast_tsne
from odin.visual import (plot_save, plot_figure, plot_scatter_heatmap,
                         plot_colorbar)

import torch

from scvi.dataset import (CortexDataset, RetinaDataset, CiteSeqDataset,
                          SyntheticRandomDataset, PbmcDataset,
                          HematoDataset, RetinaDataset)
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
from sisua.data.const import MARKER_GENES

from sisua.label_threshold import GMMThresholding
from sisua.inference import Inference
from sisua.analysis.imputation_benchmarks import (
    imputation_score, imputation_mean_score, plot_imputation,
    get_imputed_indices, plot_marker_genes, ProteinGeneAnalysis)
from sisua.analysis.latent_benchmarks import (
    streamline_classifier, plot_latents, plot_distance_heatmap,
    plot_latents_multiclasses)
from sisua.utils.io_utils import save_data
from sisua.utils.sisua_to_scvi import FacsDataset, PbmcCiteseqDataset

# ===========================================================================
# Configurations
# ===========================================================================
SAVE_DATA_PATH = "/mnt/sda1/bio_data"
FIGURE_PATH = "/tmp"
TRAIN_SIZE = 0.75
BATCH_SIZE = 64
LEARNING_RATE = 0.001
n_samples_tsne = 1000

# ====== network configuration ====== #
n_hidden = 128
n_latent = 10
n_layers = 1
dropout_rate = 0.1

# ====== SEMI_SUPERVISED learning config ====== #
SEMI1_NAME = 'dovae'
SEMI2_NAME = 'dlvae'
SEMI_PERCENT = 0.8
SISUA_XNORM = 'raw'

# ====== others ====== #
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RAND = np.random.RandomState(seed=52181208)
# ===========================================================================
# Helpers
# ===========================================================================
def to_tensor(x):
  return torch.from_numpy(x.astype('float32')).to(torch_device)

def to_array(x):
  if hasattr(x, 'todense'):
    return np.array(x.todense())
  if hasattr(x, 'cpu'):
    return x.data.cpu().numpy()
  return x

def draw_custom_hist_data(x, y):
  # First digitize the original values
  # makes a K x K matrix
  K = 15
  hist, count_bins = np.histogram(mapping(y), bins=K)
  y_bin_assign = np.digitize(mapping(y), count_bins)

  # now we get for each value and for each eventual posterior the histogram assignment
  x_bin_assign = np.digitize(mapping(x), count_bins)

  #print np.unique(y_bin_assign)
  # Let us construct the full matrix to visualize

  mat = np.zeros(shape=(K, K))
  for k in range(1, K + 1):
    assign_slice = x_bin_assign[y_bin_assign == k].flatten()
    assign_slice = assign_slice[assign_slice > 0]
    assign_slice = assign_slice[assign_slice < K + 1]
    for i in range(1, K + 1):
      assign_slice = np.append(assign_slice, i)
    #print np.unique(assign_slice)
    _, mat[k - 1] = np.unique(assign_slice, return_counts=True)

  #print mat
  mat = mat / np.sum(mat, axis=0)
  plt.xticks(np.arange(mat.shape[0])[::2], inv_mapping(count_bins).astype(np.int)[::2])
  plt.yticks(np.arange(mat.shape[0])[::2], inv_mapping(count_bins).astype(np.int)[::2])
  plt.imshow(mat.T, origin='lower', interpolation="none")

def show_distance_heatmap(Z_, y, labels_name):
  y_train, y_test = y
  n_models = len(Z_)

  plot_figure(nrow=9, ncol=5 * n_models)
  for i, (name, (ztrain, ztest)) in enumerate(Z_.items()):
    plot_distance_heatmap(ztrain, labels=y_train, labels_name=labels_name,
                          legend_enable=True if i == 0 else False,
                          ax=(2, n_models, i + 1), fontsize=10, legend_ncol=2,
                          title="%s-train" % name)
    plot_distance_heatmap(ztest, labels=y_test, labels_name=labels_name,
                          legend_enable=False,
                          ax =(2, n_models, (i + 1) + n_models),
                          title="%s-test" % name)

# ===========================================================================
# Base test
# ===========================================================================
def base_test(gene_dataset,
              n_epoch=128, use_PCA=False, train_model=True,
              corrupted_rate=0.25, corrupted_dist='uniform',
              save_outputs=None, save_figures=None):
  """
  corrupted_dist : {'uniform', 'binomial'}
  """
  # ====== training the scVI network ====== #
  vae = VAE(gene_dataset.nb_genes, n_batch=0,
            n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
            dropout_rate=dropout_rate)
  trainer = UnsupervisedTrainer(vae, gene_dataset,
    train_size=TRAIN_SIZE, use_cuda=torch.cuda.is_available(), frequency=1,
    data_loader_kwargs=dict(batch_size=BATCH_SIZE))
  trainer.corrupt_posteriors(rate=corrupted_rate, corruption=corrupted_dist)
  if train_model:
    start_time = time.time()
    trainer.train(n_epochs=n_epoch, lr=LEARNING_RATE)
    print("Finish [scVI] training: %.2f (s)" % (time.time() - start_time))
  trainer.uncorrupt_posteriors()

  # ====== get data from scVI ====== #
  train_ids = trainer.train_set.indices
  test_ids = trainer.test_set.indices

  X_train, y_train = trainer.train_set.raw_data()
  X_test, y_test = trainer.test_set.raw_data()
  X_train_corrupt = gene_dataset.corrupted_X[train_ids]
  X_test_corrupt = gene_dataset.corrupted_X[test_ids]

  # make sure convert sparse matrix to dense matrix
  X_train = to_array(X_train)
  X_test = to_array(X_test)
  X_train_corrupt = to_array(X_train_corrupt)
  X_test_corrupt = to_array(X_test_corrupt)

  # find all imputed cells
  imp_cell_train = get_imputed_indices(X_train, X_train_corrupt)
  imp_cell_test = get_imputed_indices(X_test, X_test_corrupt)

  assert not np.all(X_train == X_train_corrupt) and \
  X_train.shape == X_train_corrupt.shape
  assert not np.all(X_test == X_test_corrupt) and \
  X_test.shape == X_test_corrupt.shape

  gene_symbols = (gene_dataset.gene_symbols
                  if hasattr(gene_dataset, 'gene_symbols') else
                  gene_dataset.gene_names)
  # ====== check which labels available ====== #
  latent_visualization_mode = None
  ### single class for each cell
  if hasattr(gene_dataset, 'cell_types'):
    latent_visualization_mode = 1
    labels_name = gene_dataset.cell_types
    y_train = one_hot(y_train, nb_classes=len(labels_name))
    y_test = one_hot(y_test, nb_classes=len(labels_name))
  ### protein marker
  elif hasattr(gene_dataset, 'adt_expression'):
    latent_visualization_mode = 2
    y_train = gene_dataset.adt_expression[train_ids]
    y_test = gene_dataset.adt_expression[test_ids]
    labels_name = gene_dataset.protein_markers
  ### only binary class ( HEMATO dataset)
  elif hasattr(gene_dataset, 'cell_types_levels'):
    latent_visualization_mode = 3
    labels_name = np.array(["Erythroblasts", "Granulocytes"])
    min_y = np.min(gene_dataset.labels)
    max_y = np.max(gene_dataset.labels)
    y_train_val = 2 * (y_train - min_y) / (max_y - min_y) - 1
    y_test_val = 2 * (y_test - min_y) / (max_y - min_y) - 1

    y_full = gene_dataset.meta.values[:, 1:]
    assert y_full.shape[1] == len(gene_dataset.cell_types_levels)
    y_train = y_full[train_ids]
    y_test = y_full[test_ids]
  else:
    raise RuntimeError("No support for gene dataset: %s" % gene_dataset)

  # ====== training the SISUA network ====== #
  sisua_config = dict(hdim=n_hidden, nlayer=n_layers, zdim=n_latent,
                      xdrop=dropout_rate, xdist='zinb', ydist='bernoulli')
  infer = Inference(
      model='vae', model_config=sisua_config,
      xnorm=SISUA_XNORM, tnorm='raw', ynorm='prob')
  semi1 = Inference(
      model=SEMI1_NAME, model_config=sisua_config,
      xnorm=SISUA_XNORM, tnorm='raw',
      ynorm='prob' if latent_visualization_mode in (2, 3) else 'raw',
      ydist='bernoulli')
  semi2 = Inference(
      model=SEMI2_NAME, model_config=sisua_config,
      xnorm=SISUA_XNORM, tnorm='raw',
      ynorm='prob' if latent_visualization_mode in (2, 3) else 'raw',
      ydist='bernoulli')
  if train_model:
    for m in (infer, semi1, semi2):
      start_time = time.time()
      m.fit(X=X_train_corrupt, y=y_train,
            supervised_percent=SEMI_PERCENT, validation_percent=0.05,
            n_mcmc_samples=1,
            batch_size=BATCH_SIZE, n_epoch=n_epoch,
            learning_rate=LEARNING_RATE)
      print(
          "Finish [%s] training: %.2f (s)" % (m.name, time.time() - start_time))

  # ====== plot learning curves ====== #
  learning_curves = OrderedDict(
      scvi=(trainer.history["ll_train_set"], trainer.history["ll_test_set"]),
      sisua=(infer.train_loss, infer.valid_loss),
      semi1=(semi1.train_loss, semi1.valid_loss),
      semi2=(semi2.train_loss, semi2.valid_loss)
  )
  plot_figure(nrow=4, ncol=len(learning_curves) * 4)
  for i, (name, (train_loss, valid_loss)) in enumerate(learning_curves.items()):
    plt.subplot(1, len(learning_curves), i + 1)
    plt.plot(train_loss, label="Train")
    plt.plot(valid_loss, label="Test")
    plt.legend()
    plt.title("%s" % name)
  plt.suptitle("Learning curve")
  plt.tight_layout()

  # ====== imputation test ====== #
  V_ = OrderedDict(
      scvi=(to_array(vae.get_sample_rate(to_tensor(X_train_corrupt), n_samples=1)),
            to_array(vae.get_sample_rate(to_tensor(X_test_corrupt), n_samples=1))),
      sisua=(infer.predict_V(X=X_train_corrupt, n_mcmc_samples=1)[0],
             infer.predict_V(X=X_test_corrupt, n_mcmc_samples=1)[0]),
      semi1=(semi1.predict_V(X=X_train_corrupt, n_mcmc_samples=1)[0],
             semi1.predict_V(X=X_test_corrupt, n_mcmc_samples=1)[0]),
      semi2=(semi2.predict_V(X=X_train_corrupt, n_mcmc_samples=1)[0],
             semi2.predict_V(X=X_test_corrupt, n_mcmc_samples=1)[0])
  )
  V_score = {name: ((imputation_score(X_train, vtrain),
                     imputation_mean_score(X_train, X_train_corrupt, vtrain)),
                    (imputation_score(X_test, vtest),
                     imputation_mean_score(X_test, X_test_corrupt, vtest)))
             for name, (vtrain, vtest) in V_.items()}

  plot_figure(nrow=8, ncol=4 * len(V_))
  for i, (name, (vtrain, vtest)) in enumerate(V_.items()):
    score_train, score_test = V_score[name]

    plot_imputation(X_train, vtrain, corrupted=X_train_corrupt,
                    ax=(2, len(V_), i + 1),
                    title=r"[%s]$d=%.2f;\bar{d}=%.2f$" % ((name,) + score_train))
    plot_imputation(X_test, vtest, corrupted=X_test_corrupt,
                    ax =(2, len(V_), (i + 1) + len(V_)),
                    title=r"[%s]$d=%.2f;\bar{d}=%.2f$" % ((name,) + score_test))
  plt.tight_layout()

  # ====== plotting the imputation correlation score ====== #
  if latent_visualization_mode == 2:
    # mapping from protein index to gene index
    marker_indices = {}
    protein_markers = labels_name.tolist()
    for prot_name, gene_name in MARKER_GENES.items():
      if prot_name in protein_markers:
        index = [i
                 for i, name in enumerate(gene_symbols)
                 if gene_name == name]
        if len(index) == 0: # still not found anything
          index = [i
                   for i, name in enumerate(gene_symbols)
                   if gene_name == name.split('_')[-1]]
        if len(index) == 1: # found
          marker_indices[protein_markers.index(prot_name)] = index[0]
    #
    corr_train = OrderedDict()
    corr_test = OrderedDict()
    # getting the spearman correlation
    for name, Xorg, Xdrop, Vorg, Yorg in [
            ('train', X_train, X_train_corrupt, OrderedDict([(i, j[0]) for i, j in V_.items()]), y_train),
            ('test', X_test, X_test_corrupt, OrderedDict([(i, j[1]) for i, j in V_.items()]), y_test)]:
      for prot_idx, gene_idx in marker_indices.items():
        prot_name = protein_markers[prot_idx]
        gene_name = gene_symbols[gene_idx]
        x_drop = Xdrop[:, gene_idx].ravel()
        x_org = Xorg[:, gene_idx].ravel()
        y = Yorg[:, prot_idx].ravel()
        #
        tmp = [spearmanr(x_org, y).correlation]
        for sys_name, v in Vorg.items():
          v = v[:, gene_idx].ravel()
          tmp.append(spearmanr(v, y).correlation)
        #
        if name == 'train':
          corr_train[prot_name + '/' + gene_name] = tmp
        else:
          corr_test[prot_name + '/' + gene_name] = tmp
    name_list = ["Original"] + list(V_.keys())
    #
    plot_figure(nrow=int(len(marker_indices) * 1.8), ncol=8)
    for i, prot_gene in enumerate(corr_train.keys()):
      i = i * 2 + 1
      trn = corr_train[prot_gene]
      tst = corr_test[prot_gene]
      ids = np.arange(len(name_list))

      plt.subplot(len(marker_indices), 2, i)
      plt.bar(ids, trn)
      for px, py in zip(ids, trn):
        plt.text(px - 0.3, 0.5 * py, '%.2f' % py)
      plt.xticks(ids, name_list)
      plt.title('[Train]%s' % prot_gene)

      plt.subplot(len(marker_indices), 2, i + 1)
      plt.bar(ids, tst)
      for px, py in zip(ids, tst):
        plt.text(px - 0.3, 0.5 * py, '%.2f' % py)
      plt.xticks(ids, name_list)
      plt.title('[Test]%s' % prot_gene)
    plt.tight_layout()
  # ====== latent space test ====== #
  for xtrain, xtest, name in ((X_train, X_test, "Original"),
                              (X_train_corrupt, X_test_corrupt, "Corrupted")):
    Z_ = OrderedDict(
        scvi=(to_array(vae.sample_from_posterior_z(x=to_tensor(xtrain), give_mean=True)),
              to_array(vae.sample_from_posterior_z(x=to_tensor(xtest), give_mean=True))),
        sisua=(infer.predict_Z(X=xtrain), infer.predict_Z(X=xtest)),
        semi1=(semi1.predict_Z(X=xtrain), semi1.predict_Z(X=xtest)),
        semi2=(semi2.predict_Z(X=xtrain), semi2.predict_Z(X=xtest)),
    )
    # simple classification
    if latent_visualization_mode == 1:
      show_distance_heatmap(Z_=Z_, y=(y_train, y_test), labels_name=labels_name)
      plt.suptitle(name)

      plot_figure(nrow=10, ncol=5 * len(Z_))
      kw = dict(labels_name=labels_name, use_PCA=use_PCA, enable_separated=False)
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        plot_latents(Z=ztrain, y=y_train, title="%s-train" % model_name,
                     show_legend=False,
                     ax=(2, len(Z_), i + 1), **kw)
        plot_latents(Z=ztest, y=y_test, title="%s-test" % model_name,
                     show_legend=True if i == 0 else False,
                     ax=(2, len(Z_), i + 1 + len(Z_)), **kw)
      plt.suptitle(name)
    # multi-classes classification
    elif latent_visualization_mode == 2:
      plot_figure(nrow=10, ncol=5 * len(Z_))
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        plot_latents_multiclasses(Z=ztrain, y=y_train, labels_name=labels_name,
                          title="[%s]%s-train" % (name, model_name),
                          use_PCA=use_PCA, ax=(2, len(Z_), i + 1))
        labs = plot_latents_multiclasses(Z=ztest, y=y_test, labels_name=labels_name,
                          title="[%s]%s-test" % (name, model_name),
                          use_PCA=use_PCA, ax=(2, len(Z_), i + 1 + len(Z_)))
      plot_colorbar('bwr', vmin=-1, vmax=1,
                    tick_location=(-1, 0, 1),
                    tick_labels=(labs[0], 'others', labs[1]))
    # 2-classes on-off
    elif latent_visualization_mode == 3:
      show_distance_heatmap(Z_=Z_,
                            y=(gene_dataset.labels[train_ids],
                               gene_dataset.labels[test_ids]),
                            labels_name=labels_name)
      plt.suptitle(name)

      plot_figure(nrow=10, ncol=5 * len(Z_))
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        plot_scatter_heatmap(x=fast_tsne(ztrain), val=y_train_val,
                             fontsize=12, ax=(2, len(Z_), i + 1),
                             title="%s-train" % model_name)
        plot_scatter_heatmap(x=fast_tsne(ztest), val=y_test_val,
                             fontsize=12, ax=(2, len(Z_), i + 1 + len(Z_)),
                             title="%s-test" % model_name)
      plot_colorbar(colormap='bwr', vmin=-1, vmax=1,
                    tick_location=(-1, 0, 1),
                    tick_labels=(labels_name[0], 'others', labels_name[1]))
      plt.suptitle(name)
    else:
      raise RuntimeError(
          "No support for this latent type, dataset: %s" % str(gene_dataset))
    # streamline classifier
    if name == "Corrupted":
      if isinstance(gene_dataset, FacsDataset):
        gmm = GMMThresholding()
        gmm.fit(np.concatenate((y_train, y_test), axis=0))
        y_train = gmm.predict(y_train)
        y_test = gmm.predict(y_test)

      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        streamline_classifier(
            Z_train=ztrain, y_train=y_train,
            Z_test=ztest, y_test=y_test,
            labels_name=labels_name,
            title='[%s]%s' % (name, model_name),
            show_plot=True)
  # ====== save all figures ====== #
  if save_figures is not None:
    save_figures = str(save_figures)
    assert '.pdf' in save_figures, \
    "Only support save PDF file but given: %s" % save_figures
    plot_save(path=save_figures, dpi=120)

  # ====== save outputs ====== #
  if save_outputs is not None:
    save_outputs = str(save_outputs)
    if not os.path.exists(save_outputs):
      os.mkdir(save_outputs)
    if os.path.isfile(save_outputs):
      raise RuntimeError(
          "save_outputs must be path to folder but given: %s" % save_outputs)

# ===========================================================================
# Main tests
# ===========================================================================
if __name__ == '__main__':
  print("Semi-1:", ctext(SEMI1_NAME, 'lightyellow'))
  print("Semi-2:", ctext(SEMI2_NAME, 'lightyellow'))
  for dist in ('binomial', 'uniform'):
    BASE_PATH = '/tmp/%s' % dist
    if not os.path.exists(BASE_PATH):
      os.mkdir(BASE_PATH)
    print("Saving results to path:", ctext(BASE_PATH, 'cyan'))

    for name, gene_dataset, n_epoch in [
        ('cortex', lambda:CortexDataset(save_path=SAVE_DATA_PATH), 250),
        # ('hemato', lambda:HematoDataset(save_path=os.path.join(SAVE_DATA_PATH, 'HEMATO/')), 250),
        ('pbmc10x', lambda:PbmcDataset(save_path=SAVE_DATA_PATH), 250),
        ('pbmc_citeseq', lambda:CiteSeqDataset(name='pbmc', save_path=SAVE_DATA_PATH), 200),
        ('facs5', lambda:FacsDataset(5), 400),
        ('facs7', lambda:FacsDataset(7), 400),
        ('cbmc', lambda:CiteSeqDataset(name='cbmc', save_path=SAVE_DATA_PATH), 200),
    ]:
      gene_dataset = gene_dataset()
      base_test(gene_dataset=gene_dataset,
                n_epoch=n_epoch,
                save_figures=os.path.join(BASE_PATH, '%s.pdf' % name),
                save_outputs=os.path.join(BASE_PATH, '%s' % name),
                corrupted_rate=0.25,
                corrupted_dist=dist,
                train_model=True)

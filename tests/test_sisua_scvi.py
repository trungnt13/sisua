# dataset.expression.index
import os
import time
from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from odin.autoconfig import get_gpu_indices
from odin.utils import one_hot, batching, ctext, select_path
from odin.ml import fast_tsne
from odin.visual import (plot_save, plot_figure, plot_scatter_heatmap,
                         plot_colorbar)

import torch

from scvi.dataset import (CortexDataset, RetinaDataset, CiteSeqDataset,
                          SyntheticRandomDataset, PbmcDataset,
                          HematoDataset, RetinaDataset)
from scvi.models import *
from scvi.inference import (UnsupervisedTrainer, SemiSupervisedTrainer,
                            JointSemiSupervisedTrainer,
                            AlternateSemiSupervisedTrainer)

from sisua.data import get_scvi_dataset
from sisua.data.const import MARKER_GENES
from sisua.label_threshold import GMMThresholding
from sisua.inference import Inference
from sisua.analysis.imputation_benchmarks import (
    imputation_score, imputation_mean_score, plot_imputation,
    get_imputed_indices, plot_marker_genes, get_correlation_scores)
from sisua.analysis.latent_benchmarks import (
    streamline_classifier, plot_latents, plot_distance_heatmap,
    plot_latents_multiclasses)
from sisua.utils.io_utils import save_data
from sisua.data.sisua_to_scvi import FacsDataset, PbmcCiteseqDataset

# ===========================================================================
# Configurations
# ===========================================================================
SAVE_DATA_PATH = select_path(
    "/mnt/sda1/bio_data",
    "/media/hdd2/trung/bio_data"
)
FIGURE_PATH = "/tmp"
TRAIN_SIZE = 0.75
BATCH_SIZE = 64
LEARNING_RATE = 0.001
n_samples_tsne = 1000

# ====== network configuration ====== #
n_hidden = 32
n_latent = 128
n_layers = 2
dropout_rate = 0.1

# ====== SEMI_SUPERVISED learning config ====== #
# SEMI_SYSTEM = ['movae', 'mlvae', 'dovae', 'dlvae']
SEMI_SYSTEM = ['movae', 'mlvae']
SEMI_PERCENT = 0.8
SISUA_XNORM = 'raw'

# ====== others ====== #
torch_device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")
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
# DCA helper
# ===========================================================================
try:
  import dca as _dca_
  from odin.autoconfig import get_session
  from keras import backend
  backend.set_session(get_session())
except ImportError as e:
  raise RuntimeError("pip install dca")
from dca.network import ZINBAutoencoder

def dca_train(X, network, learning_rate=0.01, epoch=8, batch_size=32):
  model = network.model
  loss = network.loss
  model.compile(loss=loss, optimizer='adam')

  inputs = {'count': X, 'size_factors': np.sum(X, axis=-1, keepdims=True)}

  loss = model.fit(inputs, X,
                   epochs=epoch,
                   batch_size=batch_size,
                   shuffle=True,
                   callbacks=[],
                   validation_split=0.05,
                   verbose=True)

  return loss

def dca_predict(network, X, mode='denoise'):
  assert mode in ('denoise', 'latent'), 'Unknown mode'
  if mode == 'denoise':
    X = network.model.predict({'count': X,
                               'size_factors': np.sum(X, axis=-1, keepdims=True)})
  elif mode == 'latent':
    X = network.encoder.predict({'count': X,
                                 'size_factors': np.sum(X, axis=-1, keepdims=True)})
  return X

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
  vae = VAE(n_input=gene_dataset.nb_genes, n_batch=0,
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

  library_train = np.sum(X_train, axis=-1)
  library_train_ids = np.argsort(library_train)

  library_test = np.sum(X_test, axis=-1)
  library_test_ids = np.argsort(library_test)

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

  # ====== training DCA networks ====== #
  dca = ZINBAutoencoder(input_size=gene_dataset.nb_genes, output_size=None,
      hidden_size=(64,) * n_layers + (n_hidden,) + (64,) * n_layers,
      input_dropout=dropout_rate,
      batchnorm=True,
      activation='relu')
  dca.build()
  dca_loss = dca_train(X=X_train_corrupt, network=dca,
            learning_rate=LEARNING_RATE, epoch=n_epoch, batch_size=BATCH_SIZE)
  dca_val_loss = dca_loss.history['val_loss']
  dca_train_loss = dca_loss.history['loss']
  # ====== training the SISUA network ====== #
  sisua_config = dict(hdim=n_hidden, nlayer=n_layers, zdim=n_latent,
                      xdrop=dropout_rate, xdist='zinb', ydist='bernoulli')
  sisua_system = [
      Inference(
          model='vae', model_config=sisua_config,
          xnorm=SISUA_XNORM, tnorm='raw', ynorm='prob'),
  ] + [Inference(
      model=name, model_config=sisua_config,
      xnorm=SISUA_XNORM, tnorm='raw',
      ynorm='prob' if latent_visualization_mode in (2, 3) else 'raw',
      ydist='bernoulli')
      for name in SEMI_SYSTEM]
  if train_model:
    for m in sisua_system:
      start_time = time.time()
      m.fit(X=X_train_corrupt, y=y_train,
            supervised_percent=SEMI_PERCENT, validation_percent=0.05,
            n_mcmc_samples=1,
            batch_size=BATCH_SIZE, n_epoch=n_epoch,
            learning_rate=LEARNING_RATE)
      print(
          "Finish [%s] training: %.2f (s)" %
          (m.name.split('_')[0], time.time() - start_time))

  # ====== plot learning curves ====== #
  learning_curves = OrderedDict(
      scvi=(trainer.history["ll_train_set"], trainer.history["ll_test_set"]),
      dca=(dca_train_loss, dca_val_loss),
  )
  for s in sisua_system:
    sname = s.name.split('_')[0]
    learning_curves[sname] = (s.train_loss, s.valid_loss)

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
      dca=(dca_predict(dca, X=X_train_corrupt, mode='denoise'),
           dca_predict(dca, X=X_test_corrupt, mode='denoise'))
  )
  for s in sisua_system:
    sname = s.name.split('_')[0]
    V_[sname] = (s.predict_V(X=X_train_corrupt, n_mcmc_samples=1)[0],
                 s.predict_V(X=X_test_corrupt, n_mcmc_samples=1)[0])

  V_score = {name: ((imputation_score(X_train, vtrain),
                     imputation_mean_score(X_train, X_train_corrupt, vtrain)),
                    (imputation_score(X_test, vtest),
                     imputation_mean_score(X_test, X_test_corrupt, vtest)))
             for name, (vtrain, vtest) in V_.items()}

  plot_figure(nrow=6, ncol=18)
  xids = np.arange(len(V_score))
  a = plt.bar(xids + 0.0, [i[0][0] for i in V_score.values()], width=0.2)
  b = plt.bar(xids + 0.2, [i[0][1] for i in V_score.values()], width=0.2)
  c = plt.bar(xids + 0.4, [i[1][0] for i in V_score.values()], width=0.2)
  d = plt.bar(xids + 0.6, [i[1][1] for i in V_score.values()], width=0.2)

  plt.xticks(xids + 0.3, list(V_score.keys()))
  plt.legend((a, b, c, d), ('[train]imputation_score',
                            '[train]imputation_mean_score',
                            '[test]imputation_score',
                            '[test]imputation_mean_score'))

  # plot_figure(nrow=8, ncol=4 * len(V_))
  # for i, (name, (vtrain, vtest)) in enumerate(V_.items()):
  #   score_train, score_test = V_score[name]

  #   plot_imputation(X_train, vtrain, corrupted=X_train_corrupt,
  #                   ax=(2, len(V_), i + 1),
  #                   title=r"[%s]$d=%.2f;\bar{d}=%.2f$" % ((name,) + score_train))
  #   plot_imputation(X_test, vtest, corrupted=X_test_corrupt,
  #                   ax =(2, len(V_), (i + 1) + len(V_)),
  #                   title=r"[%s]$d=%.2f;\bar{d}=%.2f$" % ((name,) + score_test))
  # plt.tight_layout()

  # ====== library size comparison ====== #
  L_ = OrderedDict(
      scvi=(
          to_array(torch.exp(vae.inference(to_tensor(X_train_corrupt), n_samples=1)[-1])).ravel(),
          to_array(torch.exp(vae.inference(to_tensor(X_test_corrupt), n_samples=1)[-1])).ravel()),
      dca = (
          np.sum(V_['dca'][0], axis=-1),
          np.sum(V_['dca'][1], axis=-1))
  )
  for s in sisua_system:
    sname = s.name.split('_')[0]
    L_[sname] = (np.sum(V_[sname][0], axis=-1),
                 np.sum(V_[sname][1], axis=-1))

  plot_figure(nrow=3 * len(L_), ncol=20)
  for nplot, (name, (l_train, l_test)) in enumerate(L_.items()):
    nplot = nplot * 2
    plt.subplot(len(L_), 2, nplot + 1)
    plt.plot(library_train[library_train_ids], linewidth=1)
    plt.plot(l_train[library_train_ids], linestyle='--', alpha=0.66, linewidth=1)
    plt.title(name + '[train]')

    plt.subplot(len(L_), 2, nplot + 2)
    plt.plot(library_test[library_test_ids], linewidth=1)
    plt.plot(l_test[library_test_ids], linestyle='--', alpha=0.66, linewidth=1)
    plt.title('[test]')

  # ====== plotting the imputation correlation score ====== #
  if latent_visualization_mode == 2:
    prot_gene_names = None
    corr_scores = OrderedDict([
        ("Original",
         (get_correlation_scores(X=X_train, y=y_train,
                                 gene_name=gene_symbols,
                                 protein_name=labels_name),
          get_correlation_scores(X=X_train, y=y_train,
                                 gene_name=gene_symbols,
                                 protein_name=labels_name)))
    ])
    for name, (V_train, V_test) in V_.items():
      scores_train = get_correlation_scores(X=V_train, y=y_train,
                                            gene_name=gene_symbols,
                                            protein_name=labels_name)
      scores_test = get_correlation_scores(X=V_test, y=y_test,
                                           gene_name=gene_symbols,
                                           protein_name=labels_name)
      corr_scores[name] = (scores_train, scores_test)
      if len(scores_train) > 0:
        assert list(scores_train) == list(scores_test)
        prot_gene_names = list(scores_train.keys())

    n_systems = len(corr_scores)
    name_list = list(corr_scores.keys())
    ids = np.arange(n_systems)
    if prot_gene_names is not None:
      plot_figure(nrow=int(len(prot_gene_names) * 1.8), ncol=12)
      for row_index, prot_gene in enumerate(prot_gene_names):
        row_index = row_index * 2 + 1
        # we only care about spearman score now
        trn = [i[prot_gene][0] for _, (i, _) in corr_scores.items()]
        tst = [i[prot_gene][0] for _, (_, i) in corr_scores.items()]
        # train scores left
        plt.subplot(len(prot_gene_names), 2, row_index)
        plt.bar(ids, trn)
        for px, py in zip(ids, trn):
          plt.text(px - 0.3, 0.5 * py, '%.2f' % py)
        plt.xticks(ids, name_list)
        plt.title('[Train]%s' % prot_gene)
        # test scores right
        plt.subplot(len(prot_gene_names), 2, row_index + 1)
        plt.bar(ids, tst)
        for px, py in zip(ids, tst):
          plt.text(px - 0.3, 0.5 * py, '%.2f' % py)
        plt.xticks(ids, name_list)
        plt.title('[Test]%s' % prot_gene)
      #
      plt.tight_layout()

  # ====== latent space test ====== #
  for xtrain, xtest, name in ((X_train, X_test, "Original"),
                              (X_train_corrupt, X_test_corrupt, "Corrupted")):
    Z_ = OrderedDict(
        scvi=(to_array(vae.sample_from_posterior_z(x=to_tensor(xtrain), give_mean=True)),
              to_array(vae.sample_from_posterior_z(x=to_tensor(xtest), give_mean=True))),
        dca=(dca_predict(dca, xtrain, 'latent'),
             dca_predict(dca, xtest, 'latent')),
    )
    for s in sisua_system:
      sname = s.name.split('_')[0]
      Z_[sname] = (s.predict_Z(X=xtrain), s.predict_Z(X=xtest))
    # simple classification
    if latent_visualization_mode == 1:
      show_distance_heatmap(Z_=Z_, y=(y_train, y_test), labels_name=labels_name)
      plt.suptitle(name)

      plot_figure(nrow=10, ncol=5 * len(Z_))
      kw = dict(labels_name=labels_name, use_PCA=use_PCA, enable_separated=False)
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        plot_latents_binary(Z=ztrain, y=y_train, title="%s-train" % model_name,
                     show_legend=False,
                     ax=(2, len(Z_), i + 1), **kw)
        plot_latents_binary(Z=ztest, y=y_test, title="%s-test" % model_name,
                     show_legend=True if i == 1 else False,
                     ax=(2, len(Z_), i + 1 + len(Z_)), **kw)
      plt.suptitle(name)
    # multi-classes classification
    elif latent_visualization_mode == 2:
      plot_figure(nrow=10, ncol=5 * len(Z_))
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        plot_latents_multiclasses(Z=ztrain, y=y_train, labels_name=labels_name,
                          title="[%s]%s-train" % (name, model_name),
                          use_PCA=use_PCA, ax=(2, len(Z_), i + 1))
        plot_latents_multiclasses(Z=ztest, y=y_test, labels_name=labels_name,
                          title="[%s]%s-test" % (name, model_name),
                          use_PCA=use_PCA, ax=(2, len(Z_), i + 1 + len(Z_)),
                          show_colorbar=True)
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

      latent_space_scores = OrderedDict()
      for i, (model_name, (ztrain, ztest)) in enumerate(Z_.items()):
        scores = streamline_classifier(
            Z_train=ztrain, y_train=y_train,
            Z_test=ztest, y_test=y_test,
            labels_name=labels_name,
            title='[%s]%s' % (name, model_name),
            show_plot=True)
        latent_space_scores[model_name] = (
            scores['F1micro'], scores['F1macro'], scores['F1weight'])

      plot_figure(nrow=6, ncol=18)
      xids = np.arange(len(latent_space_scores))
      a = plt.bar(xids, [i[0] for i in latent_space_scores.values()], width=0.2)
      b = plt.bar(xids + 0.2, [i[1] for i in latent_space_scores.values()], width=0.2)
      c = plt.bar(xids + 0.4, [i[2] for i in latent_space_scores.values()], width=0.2)
      plt.xticks(xids + 0.2, list(latent_space_scores.keys()))
      plt.legend((a, b, c), ('F1-micro', 'F1-macro', 'F1-weighted'))
      plt.ylabel("F1")

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
  for dist in ('binomial', 'uniform'):
    BASE_PATH = '/tmp/%s' % dist
    if not os.path.exists(BASE_PATH):
      os.mkdir(BASE_PATH)
    print("Saving results to path:", ctext(BASE_PATH, 'cyan'))

    for name, gene_dataset, n_epoch in [
        # ('cortex', lambda:CortexDataset(save_path=SAVE_DATA_PATH), 200),
        # ('hemato', lambda:HematoDataset(save_path=os.path.join(SAVE_DATA_PATH, 'HEMATO/')), 250),
        # ('pbmc10x', lambda:PbmcDataset(save_path=SAVE_DATA_PATH), 250),
        # ('pbmc_ly', lambda:get_scvi_dataset('pbmc_ly'), 180),
        ('pbmc_citeseq', lambda:CiteSeqDataset(name='pbmc', save_path=SAVE_DATA_PATH), 1),
        # ('facs5', lambda:FacsDataset(5), 400),
        # ('facs7', lambda:FacsDataset(7), 400),
        # ('cbmc', lambda:CiteSeqDataset(name='cbmc', save_path=SAVE_DATA_PATH), 200),
    ]:
      gene_dataset = gene_dataset()
      base_test(gene_dataset=gene_dataset,
                n_epoch=n_epoch,
                save_figures=os.path.join(BASE_PATH, '%s.pdf' % name),
                save_outputs=os.path.join(BASE_PATH, '%s' % name),
                corrupted_rate=0.25,
                corrupted_dist=dist,
                train_model=True)

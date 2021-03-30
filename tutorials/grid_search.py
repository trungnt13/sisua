from __future__ import absolute_import, division, print_function

import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from odin.utils import ArgController, stdio
from odin.utils.mpi import MPI
from sisua.analysis import Posterior
from sisua.data import get_dataset
from sisua.models.autoencoder import DeepCountAutoencoder
from sisua.models.scvi_models import SCVI
from sisua.models.semi_supervised import MultitaskAutoEncoder, multitaskVAE
from sisua.models.variational_autoencoder import VariationalAutoEncoder

# turn off TF logging and set reproducibile random seed
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(8)
np.random.seed(8)

x, y = get_dataset('pbmc8kly')
x_train, x_test = x.split()
y_train, y_test = y.split()
x_train.assert_matching_cells(y_train)
x_test.assert_matching_cells(y_test)

flags = ArgController().add('--no-train', 'Stop training',
                            False).add('--no-score', 'Stop scoring',
                                       False).add('--analyze', "Analyzing",
                                                  False).parse()
no_train = flags.no_train
no_score = flags.no_score
analyze = flags.analyze
# assume the scores were ready when analyze is enable
if analyze:
  no_train = True
  no_score = True

# ===========================================================================
# Configurations
# ===========================================================================
path = '/tmp/grid'
if not os.path.exists(path):
  os.mkdir(path)
stdio(os.path.join(path, 'log.txt'))

gene = x_train.shape[1]
prot = y_train.shape[1]
epochs = 200
batch_size = 128
ncpu = 4

# ===========================================================================
# Generate all jobs
# ===========================================================================
jobs = []
for nlayers in [1, 2, 3]:
  for hdim in [32, 128, 512]:
    for zdim in [16, 32, 64]:
      for model in [
          SCVI, DeepCountAutoencoder, VariationalAutoEncoder, multitaskVAE
      ]:
        jobs.append((nlayers, hdim, zdim, model))


# ===========================================================================
# Run the training multiprocessing
# ===========================================================================
def run_training(args):
  n, h, z, model = args
  kw = dict(nlayers=n, hdim=h, zdim=z)
  name = model.id + '_%d_%d_%d' % (n, h, z)
  if model == multitaskVAE:
    model = model((gene, prot), **kw)
  else:
    model = model(gene, **kw)

  start_time = time.time()
  try:
    model.fit((x_train, y_train) if model.is_semi_supervised else x_train,
              epochs=epochs,
              batch_size=batch_size,
              semi_weight=10,
              verbose=False)
  except Exception as e:
    print("Error:", e)
    print("Error Config:", name)
    return

  print("Finish training %-4s layer:%d hdim:%-3d zdim:%d in %.2f(s)" %
        (model.id, n, h, z, time.time() - start_time))
  with open(os.path.join(path, name), 'wb') as f:
    pickle.dump(model, f)


if not no_train:
  mpi = MPI(jobs=jobs, func=run_training, ncpu=ncpu, batch=1)
  for i, j in enumerate(mpi):
    if i % 5 == 0:
      print(" == Training %d/%d jobs ==" % (i + 1, len(jobs)))


# ===========================================================================
# Generate scores file for all model
# ===========================================================================
def run_scoring(args):
  n, h, z, model = args
  name = model.id + '_%d_%d_%d' % (n, h, z)
  with open(os.path.join(path, name), 'rb') as f:
    model = pickle.load(f)

    start_time = time.time()
    post = Posterior(model,
                     x_test,
                     y_test,
                     batch_size=16,
                     sample_shape=10,
                     verbose=False)
    pearson = np.mean(list(post.scores_pearson().values()))
    spearman = np.mean(list(post.scores_spearman().values()))
    cluster = np.mean(list(post.scores_clustering().values()))
    llk = post.scores_llk()
    llk_imputed = llk['llk_imputed']
    llk = llk['llk']
    print("Scored %-4s layer:%d hdim:%-3d zdim:%d in %.2f(s)" %
          (model.id, n, h, z, time.time() - start_time))
  scores = dict(layer=n,
                hdim=h,
                zdim=z,
                model=model.id,
                pearson=pearson,
                spearman=spearman,
                llk=llk,
                llk_imputed=llk_imputed,
                cluster=cluster)
  return scores


if not no_score:
  mpi = MPI(jobs=jobs, func=run_scoring, ncpu=2, batch=1)
  all_scores = []
  for i, scores in enumerate(mpi):
    if i % 5 == 0:
      print(" == Scoring %d/%d jobs ==" % (i + 1, len(jobs)))
    all_scores.append(scores)

  df = pd.DataFrame(all_scores)
  with open(os.path.join(path, 'scores'), 'wb') as f:
    pickle.dump(df, f)

# ===========================================================================
# Finally the analysis
# ===========================================================================
if analyze:
  with open(os.path.join(path, 'scores'), 'rb') as f:
    scores: pd.DataFrame = pickle.load(f)
  g = scores.groupby(by=['model', 'zdim'])

  model = ['dca', 'scvi', 'vae', 'mvae']
  zdim = sorted(set([i[1] for i in g.groups.keys()]))
  n_row = len(zdim)
  n_col = len(model)

  plt.figure(figsize=(8, 5.5))
  plot_idx = 1
  for z in zdim:  # each row
    for m in model:  # each column
      ids = g.groups[(m, z)]
      tab = scores.iloc[ids]

      tab = tab.sort_values(by=['pearson', 'spearman'], axis=0)
      pearson = tab.pivot("layer", "hdim", "pearson")
      spearman = tab.pivot("layer", "hdim", "spearman")
      correlation = (pearson + spearman) / 2

      # tab = tab.sort_values(by='llk', axis=0)
      # correlation =  tab.pivot("layer", "hdim", "llk")

      ax = plt.subplot(n_row, n_col, plot_idx)
      sns.heatmap(correlation,
                  annot=True,
                  cmap='Blues',
                  cbar=False,
                  xticklabels="auto",
                  yticklabels="auto")
      ax.set_xlabel(None)
      ax.set_ylabel(None)
      # plt.title('%s-%d' % (m, z))
      plot_idx += 1
  plt.tight_layout()
  plot_save(os.path.join(path, 'compare.pdf'))

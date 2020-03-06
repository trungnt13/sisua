from __future__ import absolute_import, division, print_function

import os
import pickle
import random
import shutil
import time

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.ml import fast_pca, fast_umap
from odin.utils import ArgController, md5_checksum
from sisua.data import OMIC, get_dataset, normalization_recipes
from sisua.models import (MISA, SCALE, SCVI, SISUA, DeepCountAutoencoder,
                          NetworkConfig, RandomVariable, VariationalAutoEncoder,
                          load, save)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sns.set()
tf.random.set_seed(8)
np.random.seed(8)

args = ArgController().add('--load', "Load and check model integrity",
                           False).parse()
BASE_DIR = '/tmp/exp'
if not os.path.exists(BASE_DIR):
  os.mkdir(BASE_DIR)

random.seed(1234)


# ===========================================================================
# Helper
# ===========================================================================
def predict2info(model, x):
  dists = tf.nest.flatten(model.predict(x, verbose=0))
  to_numbers = lambda d: [
      fn(i).numpy()
      for i in (d.mean(), d.variance())
      for fn in (tf.reduce_mean, tf.reduce_min, tf.reduce_max)
  ] + [j for i in (d.mean(), d.variance()) for j in i.numpy().ravel()[::20]]
  text = sorted([i for d in dists for i in to_numbers(d)])
  return text, dists


def clean_folder(path, filter=None):
  if os.path.exists(path) and os.path.isdir(path):
    for name in os.listdir(path):
      f = os.path.join(path, name)
      # filtering
      if filter is not None and callable(filter):
        if not filter(f):
          continue
      # remove
      if os.path.isfile(f):
        os.remove(f)
      elif os.path.isdir(f):
        shutil.rmtree(f)


def get_weight_md5(model):
  return '.'.join(
      [md5_checksum(w) if w.ndim > 0 else str(w) for w in model.get_weights()])


def squeeze(x):
  if hasattr(x, 'numpy'):
    x = x.numpy()
  if x.shape[0] == 1:
    x = np.squeeze(x, axis=0)
  return x


def extract_pca(p_train, p_test):
  # p_train, p_test : the output and latent distributions
  pca = [
      fast_pca(squeeze(train.mean()), squeeze(test.mean()), n_components=2)[-1]
      for train, test in zip(p_train, p_test)
      if train.event_shape[0] > 1
  ]
  return pca


# ===========================================================================
# Load data
# ===========================================================================
sco = get_dataset('cortex')
train, test = sco.split(train_percent=0.8, seed=1)
n_gene = sco.numpy(OMIC.transcriptomic).shape[1]
n_prot = sco.numpy(OMIC.celltype).shape[1]

gene_rv = RandomVariable(n_gene, 'zinb', 'rna')
prot_rv = RandomVariable(n_prot, 'nb', 'adt')
latent_dim = 10
all_models = [SCALE, SCVI, DeepCountAutoencoder, VariationalAutoEncoder]
all_configs = [
    NetworkConfig(),
    NetworkConfig(pyramid=True),
    NetworkConfig(use_conv=True),
    NetworkConfig(pyramid=True, use_conv=True)
]

# ===========================================================================
# Train network
# ===========================================================================
# clean all the analysis pdf file
clean_folder(BASE_DIR, lambda f: '.pdf' == f[-4:])

for MODEL in all_models:
  for network in all_configs:
    for is_semi in (True, False):
      # ====== prepare the paths ====== #
      path = os.path.join(
          BASE_DIR,
          '%s_%s_%s_%s' % \
            (MODEL.id, 'cnn' if network.use_conv else 'dnn',
             'pyra' if network.pyramid else 'flat',
                'semi' if is_semi else 'unsp'))
      print("\n%s:" % ("Load" if args['load'] else "Save"), path)
      if not os.path.exists(path):
        os.mkdir(path)
      log_path = os.path.join(path, 'log.pkl')
      pca_path = path + '_pca.pdf'
      stat_path = path + '_stat.pdf'
      hist_path = path + '_hist.pdf'
      # ====== loading mode ====== #
      if bool(args['load']):
        with open(log_path, 'rb') as f:
          log = pickle.load(f)
        model = load(path)
        # assert model.summary() == log['summary'], "Summary mismatch"
        # assert get_weight_md5(model) == log['weight_md5'], "Weights mismatch"
        # test statistics
        plt.figure(figsize=(12, 5))
        text_train, p_train = predict2info(model, x_train)
        text_test, p_test = predict2info(model, x_test)
        plt.subplot(1, 2, 1)
        plt.plot(tf.math.log(text_train), label='Loaded')
        plt.plot(tf.math.log(log['predict_train']), label='Saved')
        plt.title("Train")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(tf.math.log(text_test), label='Loaded')
        plt.plot(tf.math.log(log['predict_test']), label='Saved')
        plt.title("Test")
        plt.legend()
        plt.tight_layout()
        vs.plot_save(stat_path, dpi=120, clear_all=True, log=True)
        # test pca
        pca = extract_pca(p_train, p_test)
        plt.figure(figsize=(8, 3 * len(pca)))
        for i, (dist, old, new) in enumerate(zip(p_train, log['pca'], pca)):
          assert old.shape == new.shape
          plt.subplot(len(pca), 2, i * 2 + 1)
          plt.scatter(old[:, 0], old[:, 1], s=4)
          if i == 0:
            plt.title('Saved')
          plt.ylabel(dist.name)

          plt.subplot(len(pca), 2, i * 2 + 2)
          plt.scatter(new[:, 0], new[:, 1], s=4)
          if i == 0:
            plt.title('Loaded')
        plt.tight_layout()
        vs.plot_save(pca_path, dpi=120, clear_all=True, log=True)
        #
        model.fit(train, epochs=2, verbose=False)
        model.plot_learning_curves()
        model.save_figures(hist_path)
      # ====== training mode ====== #
      else:
        clean_folder(path)
        epochs = int(np.random.randint(5, 15, size=1))
        model = MODEL(outputs=[gene_rv, prot_rv] if is_semi else gene_rv,
                      latent_dim=10,
                      network=network)
        start_time = time.time()
        model.fit(train,
                  epochs=epochs,
                  verbose=False)
        print(" Train   %.2f (sec/epoch)" %
              ((time.time() - start_time) / epochs))
        #
        start_time = time.time()
        text_train, p_train = predict2info(model, x_train)
        text_test, p_test = predict2info(model, x_test)
        print(" Predict %.2f (sec)" % (time.time() - start_time))
        pca = extract_pca(p_train, p_test)
        #
        log = dict(summary=model.summary(),
                   predict_train=text_train,
                   predict_test=text_test,
                   pca=pca,
                   weight_md5=get_weight_md5(model))
        #
        with open(log_path, 'wb') as f:
          pickle.dump(log, f)
        save(path, model)

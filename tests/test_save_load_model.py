from __future__ import absolute_import, division, print_function

import os
import pickle
import random
import time
import unittest
import warnings
from tempfile import mkstemp

import numpy as np
import tensorflow as tf

from odin.ml import fast_pca
from odin.utils import catch_warnings_ignore, md5_checksum
from sisua.data import OMIC, get_dataset, normalization_recipes
from sisua.models import (MISA, SCALE, SCVI, SISUA, DeepCountAutoencoder,
                          NetworkConfig, RandomVariable, VariationalAutoEncoder,
                          load, save)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)


# ===========================================================================
# Helpers
# ===========================================================================
def predict2info(model, x):
  pX, qZ = model.predict(x, verbose=0)
  pX = tf.nest.flatten(pX)
  qZ = tf.nest.flatten(qZ)
  dists = pX + qZ
  to_numbers = lambda d: [
      fn(i).numpy()
      for i in (d.mean(), d.variance())
      for fn in (tf.reduce_mean, tf.reduce_min, tf.reduce_max)
  ] + [j for i in (d.mean(), d.variance()) for j in i.numpy().ravel()[::20]]
  text = sorted([i for d in dists for i in to_numbers(d)])
  # the latent space likely to be the same after loading
  return text, dists, \
    qZ[0].mean().numpy().astype(np.float64), \
      qZ[0].variance().numpy().astype(np.float64)


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


def model_iteration(models, configs):
  for MODEL in models:
    for network in configs:
      for is_semi in (True, False):
        path = os.path.join(
            BASE_DIR,
            '%s_%s_%s_%s' % \
              (MODEL.id, 'cnn' if network.use_conv else 'dnn',
               'pyra' if network.pyramid else 'flat',
                  'semi' if is_semi else 'unsp'))
        if not os.path.exists(path):
          os.mkdir(path)
        log_path = os.path.join(path, 'log.pkl')
        pca_path = path + '_pca.pdf'
        stat_path = path + '_stat.pdf'
        hist_path = path + '_hist.pdf'
        print("%s" % path)
        yield (MODEL, network, is_semi, path, log_path, pca_path, stat_path,
               hist_path)


# ===========================================================================
# Main Test
# ===========================================================================
BASE_DIR = '/tmp/sisua_test'
if not os.path.exists(BASE_DIR):
  os.mkdir(BASE_DIR)
LATENT_DIM = 10


class SaveLoadModelTest(unittest.TestCase):

  def prepare(self):
    with catch_warnings_ignore(RuntimeWarning):
      sco = get_dataset('cortex')
      om1, om2 = sco.omics
      train, test = sco.split(train_percent=0.8, seed=1)
      n_gene = sco.numpy(om1).shape[1]
      n_prot = sco.numpy(om2).shape[1]
      rvs = [
          RandomVariable(n_gene, 'zinbd', om1.name),
          RandomVariable(n_prot, 'onehot', om2.name)
      ]
      all_models = [DeepCountAutoencoder, SCALE, SCVI, VariationalAutoEncoder]
      all_configs = [
          NetworkConfig(),
          NetworkConfig(pyramid=True),
          NetworkConfig(use_conv=True),
          NetworkConfig(pyramid=True, use_conv=True)
      ]
      return train, test, rvs, all_models, all_configs

  ###
  def test_save_load_1(self):
    r""" Train and save the model """
    print("*** Test saving model ***")
    train, test, rvs, models, configs = self.prepare()
    for (MODEL, network, is_semi, path, log_path, pca_path, stat_path,
         hist_path) in model_iteration(models, configs):
      clean_folder(path)
      epochs = int(np.random.randint(5, 15, size=1))
      model = MODEL(outputs=rvs if is_semi else rvs[0],
                    latent_dim=10,
                    network=network)
      start_time = time.time()
      model.fit(train, epochs=epochs, verbose=False)
      print("  Train   %.2f (sec/epoch)" %
            ((time.time() - start_time) / epochs))
      #
      start_time = time.time()
      text_train, p_train, zmean_train, zvar_train = predict2info(model, train)
      text_test, p_test, zmean_test, zvar_test = predict2info(model, test)
      print("  Predict %.2f (sec)" % (time.time() - start_time))
      pca = extract_pca(p_train, p_test)
      #
      log = dict(summary=model.summary(),
                 predict_train=text_train,
                 predict_test=text_test,
                 zmean=(zmean_train, zmean_test),
                 zvar=(zvar_train, zvar_test),
                 pca=pca,
                 weight_md5=get_weight_md5(model))
      #
      with open(log_path, 'wb') as f:
        pickle.dump(log, f)
      save(path, model)

  ###
  def test_save_load_2(self):
    r""" Load and train the model """
    print("*** Test loading model ***")
    from matplotlib import pyplot as plt
    from odin import visual as vs
    import seaborn as sns
    sns.set()
    #
    train, test, rvs, models, configs = self.prepare()
    for (MODEL, network, is_semi, path, log_path, pca_path, stat_path,
         hist_path) in model_iteration(models, configs):
      with open(log_path, 'rb') as f:
        log = pickle.load(f)
      model = load(path)
      with catch_warnings_ignore(UserWarning):
        # test statistics
        plt.figure(figsize=(12, 5))
        text_train, p_train, zmean_train, zvar_train = predict2info(
            model, train)
        text_test, p_test, zmean_test, zvar_test = predict2info(model, test)
        # check latent mean and variance
        zmean_train1, zmean_test1 = log['zmean']
        zvar_train1, zvar_test1 = log['zvar']
        self.assertTrue(np.allclose(zmean_train, zmean_train1))
        self.assertTrue(np.allclose(zmean_test, zmean_test1))
        self.assertTrue(np.allclose(zvar_train, zvar_train1))
        self.assertTrue(np.allclose(zvar_test, zvar_test1))
        # plotting
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
          #
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


if __name__ == '__main__':
  unittest.main()

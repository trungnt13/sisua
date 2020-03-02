from __future__ import absolute_import, division, print_function

import itertools
import os
import unittest
from tempfile import mkstemp

import numpy as np
import pandas as pd

from odin.utils import catch_warnings_ignore
from sisua.data import OMIC, get_dataset

np.random.seed(8)


# ===========================================================================
# Helpers
# ===========================================================================
def _equal(self, sco1, sco2):
  self.assertTrue(np.all(sco1.X == sco2.X))
  for (k1, v1), (k2, v2) in zip(\
    itertools.chain(sco1.obs.items(), sco1.obsm.items()),
    itertools.chain(sco2.obs.items(), sco2.obsm.items())):
    self.assertEqual(k1, k2)
    self.assertTrue(np.all(v1 == v2), msg="obs-key: %s" % k1)
  for (k1, v1), (k2, v2) in zip(\
    itertools.chain(sco1.var.items(), sco1.varm.items()),
    itertools.chain(sco2.var.items(), sco2.varm.items())):
    self.assertEqual(k1, k2)
    self.assertTrue(np.all(v1 == v2), msg="var-key: %s" % k1)
  for (k1, v1), (k2, v2) in zip(sco1.uns.items(), sco2.uns.items()):
    self.assertEqual(k1, k2)
    self.assertTrue(type(v1) is type(v2))
    if isinstance(v1, np.ndarray):
      cond = np.all(v1 == v2)
    elif isinstance(v1, pd.DataFrame):
      cond = np.all(v1 == v2) and np.all(v1.index == v2.index)
    else:
      cond = v1 is v2
    self.assertTrue(cond, msg="uns-key: %s" % k1)


# ===========================================================================
# Main Tests
# ===========================================================================
class SisuaDataset(unittest.TestCase):

  def test_basic_functionalities(self):
    ds = get_dataset('8kmy')
    # split
    train, test = ds.split()
    self.assertEqual(set(train.cell_id) | set(test.cell_id), set(ds.cell_id))
    # copy
    copy1 = ds.copy()  # copy backed dataset
    copy2 = train.copy()  # copy view dataset
    copy3 = ds.copy().apply_indices(test.indices)
    _equal(self, copy1, ds)
    _equal(self, copy2, train)
    _equal(self, copy3, test)
    # split again
    train1, test1 = ds.split()
    train.assert_matching_cells(train1)
    test.assert_matching_cells(test1)
    _equal(self, train, train1)
    _equal(self, test, test1)

  def test_corruption(self):
    ds = get_dataset('8kmy')
    ds1 = ds.corrupt(dropout_rate=0.25, inplace=False)
    ds2 = ds.corrupt(dropout_rate=0.5, inplace=False)
    ds3 = ds.corrupt(dropout_rate=0.5, inplace=False, omic=OMIC.proteomic)
    ds4 = ds.corrupt(dropout_rate=0.5,
                     inplace=False,
                     omic=OMIC.proteomic,
                     distribution='uniform')
    self.assertTrue(ds.sparsity() < ds1.sparsity() < ds2.sparsity())
    om = OMIC.proteomic
    self.assertTrue(ds.sparsity(om) < ds3.sparsity(om) < ds4.sparsity(om))

  def test_filters(self):
    ds = get_dataset('8kmy')
    ds1 = ds.filter_highly_variable_genes(inplace=False)
    ds2 = ds.filter_genes(inplace=False, min_counts=100)
    ds3 = ds.filter_cells(inplace=False, min_counts=1000)
    self.assertTrue(ds1.shape[1] == 999)
    self.assertTrue(np.min(ds2.X.sum(0)) == 100)
    self.assertTrue(np.min(ds3.X.sum(1)) == 1000)

  def test_embedding(self):
    ds = get_dataset('8kmy')
    ds.probabilistic_embedding()
    prob = ds.probability()
    bina = ds.binary()
    self.assertTrue(np.all(np.logical_and(0. < prob, prob < 1.)))
    self.assertTrue(np.all(np.unique(bina) == np.unique([0., 1.])))

    for algo in ('pca', 'tsne'):
      n = ds.n_obs
      pca1 = ds.dimension_reduce(n_components=2, algo=algo)
      pca2 = ds.dimension_reduce(OMIC.proteomic, n_components=3, algo=algo)
      self.assertTrue(pca1.shape == (n, 2))
      self.assertTrue(pca2.shape == (n, 3) if algo == 'pca' else \
        pca2.shape == (n, 2))
      name1 = '%s_%s' % (OMIC.proteomic.name, algo)
      name2 = '%s_%s' % (OMIC.transcriptomic.name, algo)
      self.assertTrue(name1 in ds.obsm and name1 in ds.uns)
      self.assertTrue(name2 in ds.obsm and name2 in ds.uns)

  def test_normalization(self):
    ds = get_dataset('8kmy')
    # ignore overflow warning
    with catch_warnings_ignore(RuntimeWarning):
      ds1 = ds.expm1(omic=OMIC.transcriptomic, inplace=False)
      ds2 = ds.expm1(omic=OMIC.proteomic, inplace=False)
      self.assertTrue(np.all(np.expm1(ds.X) == ds1.X))
      self.assertTrue(
          np.all(
              np.expm1(ds.numpy(OMIC.proteomic)) == ds2.numpy(OMIC.proteomic)))

    ds1 = ds.normalize(OMIC.transcriptomic,
                       inplace=False,
                       log1p=True,
                       scale=False,
                       total=False)
    ds2 = ds.normalize(OMIC.proteomic,
                       inplace=False,
                       log1p=True,
                       scale=False,
                       total=False)
    self.assertTrue(np.all(ds1.numpy(OMIC.transcriptomic) == np.log1p(ds.X)))
    self.assertTrue(
        np.all(ds1.numpy(OMIC.proteomic) == ds.numpy(OMIC.proteomic)))
    self.assertTrue(
        np.all(ds2.numpy(OMIC.proteomic) == np.log1p(ds.numpy(OMIC.proteomic))))
    self.assertTrue(
        np.all(ds2.numpy(OMIC.transcriptomic) == ds.numpy(OMIC.transcriptomic)))

  def test_metrics(self):
    ds = get_dataset('8kmy')
    ds.calculate_quality_metrics()
    ds.rank_genes_groups()

  def test_clustering(self):
    ds = get_dataset('8kmy')
    # ds.kmeans()
    # ds.knn()


if __name__ == '__main__':
  unittest.main()

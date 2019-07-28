from __future__ import print_function, division, absolute_import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from sisua import data
from sisua.data import SingleCellOMICS
from sisua.models.variational_autoencoder import (
  VariationalAutoencoder)
import scanpy as sc

from odin.fuel import MmapArray
from odin.utils import UnitTimer

x, y = data.get_dataset('pbmc8k_ly', override=False)

y.probabilistic_embedding()
m, n = y.split()
print(y)
print(m)
print(n)

y, z = x.split()
w = x.corrupt(inplace=False)
a = x.filter_highly_variable_genes(inplace=False)
b = x.filter_genes(min_counts=50, inplace=False)
c = x.filter_cells(min_counts=500, inplace=False)
d = x.normalize(total_counts=True, log1p=True, scale=True, inplace=False)
x.calculate_qc_metrics()
# y.probabilistic_embedding()

assert np.all(m.obs.iloc[:, 0].values ==  y.obs.iloc[:, 0].values)
assert np.all(n.obs.iloc[:, 0].values ==  z.obs.iloc[:, 0].values)

print(x)
print(y)
print(z)
print(w)
print(a)
print(b)
print(c)
print(d)

x.save_to_mmaparray('/tmp/array', dtype='float32')
x.plot_percentile_histogram(n_hist=12).save_figures('/tmp/tmp.pdf')
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from sisua.data import get_dataset, normalization_recipes
from sisua.models import DeepCountAutoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

x, y = get_dataset('pbmc8kly')
x_train, x_test = x.split()
y_train, y_test = y.split()
x_train.assert_matching_cells(y_train)
x_test.assert_matching_cells(y_test)

n_genes = x_train.shape[1]
n_prots = y_train.shape[1]

dca = DeepCountAutoencoder(units=n_genes, xdist='zinb')
dca.fit(x_train, epochs=8)

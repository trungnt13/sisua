from __future__ import print_function, division, absolute_import

import tensorflow as tf

from odin import nnet as N, backend as K
from odin.utils import get_script_path

from sisua import set_verbose
from sisua.utils.sisua_to_scvi import FacsDataset
from sisua.inference import Inference

# ===========================================================================
# Load the dataset
# ===========================================================================
ds = FacsDataset()
X = ds.X
y = ds.adt_expression

# ===========================================================================
# Define model as a function
# ===========================================================================
# This decorator is important, it converts everything to O.D.I.N
# NNOp
@N.Lambda
def network_keras(X, T, C, mask, y, nsample, kwargs):
  """ In this example we define the model using keras
  For more information please check the file:
  'models_new.py' in the 'tutorials' folder
  """
  if kwargs['tnorm'] == 'raw':
    output_activation = 'relu'
  elif kwargs['tnorm'] == 'log':
    output_activation = 'softplus'
  else:
    output_activation = 'softmax'
  X_dim = X.shape.as_list()[-1]

  # you have to make sure the input_tensor is X in keras
  network = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_tensor=X),
      tf.keras.layers.Dense(units=32, activation='relu', name="EncoderLayer1"),
      tf.keras.layers.Dense(units=32, activation='relu', name="EncoderLayer2"),

      tf.keras.layers.Dense(units=kwargs['zdim'], use_bias=False,
                            activation='linear', name="Bottleneck"),

      tf.keras.layers.Dense(units=32, activation='relu', name="DecoderLayer1"),
      tf.keras.layers.Dense(units=32, activation='relu', name="DecoderLayer2"),

      tf.keras.layers.Dense(units=X_dim, activation=output_activation,
                            name="OutputLayer")
  ], name="SingleCellNetwork")

  W = network(X)
  loss = tf.losses.mean_squared_error(labels=T, predictions=W)
  Z = network.get_layer("Bottleneck").output
  return {'W': W, 'Z': Z, 'loss': loss,
          'encoder': network.get_layer("EncoderLayer1").output}

# ===========================================================================
# Train the network
# ===========================================================================
# Turn verbose True if you want some detail logging of the model
set_verbose(False)
n_epoch = 25

# model can be defined from separated file, and given
# by specifying the `extra_module_path`
infer1 = Inference(model='newmodel1',
                  extra_module_path=get_script_path())
infer1.fit(X, n_epoch=n_epoch)
print('Model ODIN:', infer1.score(X))

# Or you can create your own NNOp object and make a model
infer2 = Inference(model=network_keras)
infer2.fit(X, n_epoch=n_epoch)
print('Model Keras:', infer2.score(X))
# ====== some plotting ====== #
infer1.plot_learning_curves(save_path='/tmp/model1.pdf')
infer2.plot_learning_curves(save_path='/tmp/model2.pdf')

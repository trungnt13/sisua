from __future__ import print_function, division, absolute_import

import tensorflow as tf
from odin import nnet as N, backend as K

@N.Lambda
def newmodel1(X, T, C, mask, y, nsample, kwargs):
  """ Following arguments a passed through `kwargs` which
  defined by user, you could choose to use them or not:
   - hdim
   - zdim
   - nlayer
   - xdist
   - ydist
   - zdist
   - xnorm
   - ynorm
   - znorm
   - batchnorm
   - xdrop
   - edrop
   - zdrop
   - ddrop
  """
  if kwargs['tnorm'] == 'raw':
    output_activation = tf.nn.relu
  elif kwargs['tnorm'] == 'log':
    output_activation = tf.nn.softplus
  else:
    output_activation = tf.softmax

  network = N.Sequence(ops=[
      N.Dropout(level=kwargs['xdrop']),
      N.Dense(num_units=32, activation=tf.nn.relu, name="EncoderLayer1"),
      N.Dense(num_units=32, activation=tf.nn.relu, name="EncoderLayer2"),

      N.Dense(num_units=kwargs['zdim'], b_init=None,
              activation=K.linear, name="Bottleneck"),

      N.Dense(num_units=32, activation=tf.nn.relu, name="DecoderLayer1"),
      N.Dense(num_units=32, activation=tf.nn.relu, name="DecoderLayer2"),

      N.Dense(num_units=X.shape.as_list()[-1], activation=output_activation,
              name="OutputLayer")
  ], debug=True, name="SingleCellNetwork")

  W = network(X)
  loss = tf.losses.mean_squared_error(labels=T, predictions=W)

  # ====== extract the latent space ====== #
  Z = K.get_intermediate_tensors(outputs=W,
                                 roles=N.Dense, scope="Bottleneck")[0]
  # ====== return ====== #
  return {'W': W, 'Z': Z, 'loss': loss}

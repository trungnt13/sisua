from __future__ import print_function, division, absolute_import

from six import add_metaclass
from abc import ABCMeta, abstractmethod

from tensorflow.python.keras.layers import Layer

@add_metaclass(ABCMeta)
class BioModel(object):
  """ BioModel """

  def __init__(self, weights=None):
    self._is_initialized = False
    self.all_weights = weights

  def get_weights(self):
    # Mistake here, BioModel could contain
    # a list or dictionary of Layer
    all_weights = {}
    for key in dir(self):
      val = getattr(self, key)
      if isinstance(val, Layer):
        all_weights[key] = ('single', val.get_weights())
      elif isinstance(val, (tuple, list)) and all(isinstance(i, Layer) for i in val):
        all_weights[key] = ('list', [i.get_weights() for i in val])
      elif isinstance(val, dict) and all(isinstance(i, Layer) for i in val.values()):
        all_weights[key] = ('dict', {i: j.get_weights() for i, j in val.items()})
    return all_weights

  def set_weights(self, weights):
    for key, (layer_type, val) in weights.items():
      if layer_type == 'list':
        layers = getattr(self, key)
        for l, v in zip(layers, val):
          l.set_weights(v)
      elif layer_type == 'dict':
        layers = getattr(self, key)
        for name, v in layers.items():
          layers[name].set_weights(v)
      elif layer_type == 'single':
        layer = getattr(self, key)
        layer.set_weights(val)
      else:
        raise RuntimeError(
            "No support for layer type '%s' of layer '%s'" % (layer_type, key))

  def __call__(self, X, T,
               L, L_mean, L_var,
               mask, y,
               nsample, nepoch, configs):
    # initialize
    if not self._is_initialized:
      self._init(X, T,
                 L, L_mean, L_var,
                 mask, y,
                 nsample, nepoch, configs)
      self._is_initialized = True
    # make the call
    output = self._call(X, T,
                        L, L_mean, L_var,
                        mask, y,
                        nsample, nepoch, configs)
    # set the weight
    if self.all_weights is not None:
      self.set_weights(self.all_weights)
      self.all_weights = None
    return output

  @abstractmethod
  def _init(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    raise NotImplementedError

  @abstractmethod
  def _call(self, X, T,
            L, L_mean, L_var,
            mask, y,
            nsample, nepoch, configs):
    raise NotImplementedError

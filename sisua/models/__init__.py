from typing import Dict, Optional, Text, Type

from sisua.models.dca import *
from sisua.models.scale import *
from sisua.models.scvi import *
from sisua.models.single_cell_model import *
from sisua.models.vae import *


def get_all_models() -> list:
  all_models = []
  for key, val in globals().items():
    if isinstance(val, type) and issubclass(val, SingleCellModel):
      all_models.append(val)
  return sorted(all_models, key=lambda cls: cls.id)


def get_model(model) -> Type[SingleCellModel]:
  if isinstance(model, type):
    model = model.__name__
  model = str(model).lower()
  for key, val in globals().items():
    if isinstance(val, type) and issubclass(val, SingleCellModel):
      if model == key.lower() or model == val.id:
        return val
  raise RuntimeError("Cannot find SingleCellModel with type '%s'" % model)


def save_model(path: str, model: SingleCellModel):
  assert isinstance(model, SingleCellModel), \
    "model must be instance of SingleCellModel but given type: %s" % \
      str(type(model))
  import os
  import pickle
  # ====== SingleCellModel arguments ====== #
  class_name = model.__class__.__name__
  kwargs = dict(model.init_args)
  # ====== save the model ====== #
  if os.path.exists(path) and os.path.isfile(path):
    raise ValueError(
        "Cannot only save model to a folder, given path to a file: %s" % path)
  if not os.path.exists(path):
    os.makedirs(path)
  model.save_weights(os.path.join(path, 'weights'))
  with open(os.path.join(path, 'arguments'), 'wb') as f:
    pickle.dump([class_name, kwargs], f)
  return path


def load_model(path: str) -> SingleCellModel:
  import os
  import pickle
  assert os.path.exists(path) and os.path.isdir(path), \
    f"{path} must be directory"
  ## create new instance
  with open(os.path.join(path, 'arguments'), 'rb') as f:
    class_name, kwargs = pickle.load(f)
  model = get_model(class_name)(**kwargs)
  ## restore the checkpoint
  model.load_weights(os.path.join(path, 'weights'), raise_notfound=True)
  return model

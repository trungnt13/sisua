from typing import Dict, Optional, Text, Type

from sisua.models.dca import *
from sisua.models.fvae import *
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


def save_model(model_dir: str, model: SingleCellModel):
  assert isinstance(model, SingleCellModel), \
    "model must be instance of SingleCellModel but given type: %s" % \
      str(type(model))
  import os
  import pickle
  # ====== SingleCellModel arguments ====== #
  class_name = model.__class__.__name__
  dataset = model.dataset
  kwargs = dict(model.init_args)
  # ====== save the model ====== #
  if os.path.exists(model_dir) and os.path.isfile(model_dir):
    raise ValueError(
        "Cannot only save model to a folder, given path to a file: %s" %
        model_dir)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  model.save_weights(os.path.join(model_dir, 'weights'))
  with open(os.path.join(model_dir, 'arguments'), 'wb') as f:
    pickle.dump([class_name, dataset, kwargs], f)
  return model_dir


def load_model(model_dir: str) -> SingleCellModel:
  import os
  import pickle
  assert os.path.exists(model_dir) and os.path.isdir(model_dir), \
    f"{model_dir} must be directory"
  ## create new instance
  with open(os.path.join(model_dir, 'arguments'), 'rb') as f:
    class_name, dataset, kwargs = pickle.load(f)
  model = get_model(class_name)(**kwargs)
  model.dataset = dataset
  ## restore the checkpoint
  model.load_weights(os.path.join(model_dir, 'weights'), raise_notfound=True)
  return model

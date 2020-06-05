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
  raise RuntimeError(f"Cannot find SingleCellModel with type '{model}'")


def load_model(filepath: str) -> SingleCellModel:
  import pickle
  ## create new instance
  with open(f"{filepath}.metamodel", 'rb') as f:
    class_name, dataset, metadata, kwargs = pickle.load(f)
  model = get_model(class_name)(**kwargs)
  ## restore the checkpoint
  model.load_weights(filepath, raise_notfound=True)
  return model

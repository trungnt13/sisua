from typing import Dict, Optional, Text

from sisua.models import latents, networks
from sisua.models.autoencoder import *
from sisua.models.base import *
from sisua.models.scscope_models import *
from sisua.models.scvi_models import *
from sisua.models.semi_supervised import *
from sisua.models.variational_autoencoder import *


def get_model(identifier: Optional[Text] = None) -> Dict:
  from sisua import models
  from types import ModuleType
  from typing import Type
  all_models = {}
  for name in dir(models):
    mod = getattr(models, name)
    if isinstance(mod, ModuleType):
      if 'sisua/models' in mod.__file__:
        for subname in dir(mod):
          clz = getattr(mod, subname)
          if isinstance(clz, Type) and issubclass(
              clz, SingleCellModel) and clz != SingleCellModel:
            all_models[clz.id] = clz
  if identifier is not None:
    return all_models[identifier]
  return all_models

from typing import Dict, Optional, Text

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


def get_model(model):
  if isinstance(model, type):
    model = model.__name__
  model = str(model).lower()
  for key, val in globals().items():
    if isinstance(val, type) and issubclass(val, SingleCellModel):
      if model == key.lower() or model == val.id:
        return val
  raise RuntimeError("Cannot find SingleCellModel with type '%s'" % model)


def save(path: str, model: SingleCellModel, max_to_keep=4):
  assert isinstance(model, SingleCellModel), \
    "model must be instance of SingleCellModel but given type: %s" % \
      str(type(model))
  from odin.backend import Trainer
  import inspect
  import pickle
  # ====== extra arguments ====== #
  args = inspect.getfullargspec(model.__class__.__init__).args[1:]
  exclude_args = set(inspect.getfullargspec(SingleCellModel.__init__).args[1:])
  extra_kwargs = {}
  for a in args:
    if a not in exclude_args and hasattr(model, a):
      attr = getattr(model, a)
      extra_kwargs[a] = attr
  # ====== SingleCellModel arguments ====== #
  class_name = model.__class__.__name__
  kwargs = dict(outputs=model._omic_outputs,
                latents=model._latents,
                network=model._network_config,
                kl_interpolate=model._kl_interpolate,
                kl_mcmc=model._kl_mcmc,
                analytic=model._analytic,
                log_norm=model._log_norm,
                seed=model._seed,
                name=model.name)
  kwargs.update(extra_kwargs)
  # ====== save the model ====== #
  if os.path.exists(path) and os.path.isfile(path):
    raise ValueError(
        "Cannot only save model to a folder, given path to a file: %s" % path)
  if not os.path.exists(path):
    os.mkdir(path)
  Trainer.save_checkpoint(path,
                          optimizers=model.optimizer,
                          models=model,
                          max_to_keep=max_to_keep)
  with open(os.path.join(path, 'history.pkl'), 'wb') as f:
    pickle.dump([model._history, model._fit_history], f)
  with open(os.path.join(path, 'singlecellmodel.pkl'), 'wb') as f:
    pickle.dump([class_name, kwargs], f)
  return path


def load(path: str, model_index=-1) -> SingleCellModel:
  assert os.path.exists(path)
  if os.path.isfile(path):
    raise ValueError(
        "Cannot only save model to a folder, given path to a file: %s" % path)
  import pickle
  from odin.backend import Trainer
  ## create new instance
  with open(os.path.join(path, 'singlecellmodel.pkl'), 'rb') as f:
    class_name, kwargs = pickle.load(f)
  model = get_model(class_name)(**kwargs)
  ## restore the checkpoint
  model, optimizer, _ = Trainer.restore_checkpoint(path,
                                                   models=model,
                                                   index=model_index)
  model = model[0]
  if len(optimizer) > 0:
    model.optimizer = optimizer[0]
  ## restore the history
  with open(os.path.join(path, 'history.pkl'), 'rb') as f:
    model._history, model._fit_history = pickle.load(f)
  ## restore the attributes
  return model

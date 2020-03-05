from __future__ import absolute_import, division, print_function

import inspect
import os
import sys

import numpy as np
import tensorflow as tf

from odin.backend import interpolation
from odin.exp import Experimenter, ExperimentManager
from odin.utils.crypto import md5_checksum
from sisua.data import (CONFIG_PATH, DATA_DIR, EXP_DIR, OMIC, SingleCellOMIC,
                        get_dataset, get_dataset_meta)
from sisua.models import (NetworkConfig, RandomVariable, get_all_models,
                          get_model, load, save)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


# ===========================================================================
# Helper
# ===========================================================================
def elbo_config(cfg):
  assert 'elbo' in cfg, \
    "Configuration must contain key: 'elbo' for elbo configuraiton "
  cfg = cfg.elbo
  itp = interpolation.get(str(cfg.get(interpolation, 'const')))
  itp = itp(vmin=cfg.get('vmin', 0.),
            vmax=cfg.get('vmax', 1.),
            norm=cfg.get('norm', 50),
            cyclical=cfg.get('cyclical', False),
            delayIn=cfg.get('delayIn', 0),
            delayOut=cfg.get('delayOut', 0))
  return dict(kl_interpolate=itp,
              kl_mcmc=int(cfg.get('mcmc', 1)),
              analytic=bool(cfg.get('analytic', True)))


def random_variable(omic: str, cfg: dict, sco: SingleCellOMIC):
  r""" Create RandomVariable from a named DictConfig

  name: the key name of the variable
  """
  name = omic.name
  if name not in cfg:
    return None
  cfg = cfg[name]
  dim = cfg.get('dim', None)
  if dim is None:
    if omic in sco.omics:
      dim = sco.numpy(omic).shape[1]
    else:  # cannot infer the dimension of the RandomVariable
      return None
  posterior = cfg.get('posterior')
  kwargs = cfg.get('kwargs', {})
  return RandomVariable(dim=int(dim), posterior=posterior, name=name, **kwargs)


# ===========================================================================
# Main class
# ===========================================================================
class SisuaExperimenter(Experimenter):

  def __init__(self, ncpu=1):
    super().__init__(save_path=EXP_DIR,
                     config_path=CONFIG_PATH,
                     exclude_keys="train",
                     ncpu=int(ncpu))

  def on_load_data(self, cfg):
    sco = get_dataset(cfg.dataset.name)
    split = float(cfg.dataset.get('split', 0.8))
    train, test = sco.split(split)
    self.train = train
    self.test = test

  def on_create_model(self, cfg):
    network = NetworkConfig(**cfg.network)
    rv_trans = None
    rv_latent = None
    rvs = []
    for om in OMIC:
      rv = random_variable(om, cfg, self.train)
      if rv is not None:
        if om == OMIC.transcriptomic:
          rv_trans = rv
        elif om == OMIC.latent:
          rv_latent = rv
        else:
          rvs.append(rv)
    log_norm = cfg.get('log_norm', True)
    force_semi = cfg.get('force_semi', False)
    if len(rvs) == 0:
      force_semi = False
    # ====== check required variable is provided ====== #
    if rv_trans is None or rv_latent is None:
      raise RuntimeError("RandomVariable description must be provided for "
                         "'transcriptomic' and 'latent'")
    # ====== create model ====== #
    model_cls = get_model(cfg.model)
    args = inspect.getfullargspec(model_cls.__init__).args
    model_kwargs = dict(network=network, log_norm=log_norm)
    model_kwargs.update(elbo_config(cfg))
    # because every model could be semi-supervised
    if model_cls.is_multiple_outputs:
      model_kwargs['rna_dim'] = rv_trans.dim
      model_kwargs['adt_dim'] = rvs[0].dim
    elif force_semi:
      model_kwargs['outputs'] = [rv_trans] + rvs
    else:
      model_kwargs['outputs'] = [rv_trans]
    if 'latent_dim' in args:
      model_kwargs['latent_dim'] = rv_latent.dim
    else:
      model_kwargs['latents'] = rv_latent
    self.model = model_cls(**model_kwargs)

  def on_load_model(self, path):
    self.model = load(path, model_index=-1)
    return self.model

  def on_train(self, cfg, model_path):
    kwargs = Experimenter.match_arguments(self.model.fit,
                                          cfg.train,
                                          exclude_args='inputs')
    self.model.fit(inputs=self.train, **kwargs)
    # save the best model after training
    save(model_path, self.model, max_to_keep=5)

  @property
  def args_help(self):
    all_models = [i.id for i in get_all_models()]
    all_datasets = list(get_dataset_meta().keys())
    return {'model': all_models, 'dataset.name': all_datasets}


# ===========================================================================
# Running the experiments
# --reset model=sisua,dca,vae dataset.name=cortex,8kly,8klyall,8klyx,eccly,ecclyall,ecclyx,8k,8kall train.epochs=500 elbo.vmax=1,2,10 -m -ncpu 5
# ===========================================================================
if __name__ == "__main__":
  print("Path:")
  print(" - exp:", EXP_DIR)
  print(" - dat:", DATA_DIR)
  print(" - cfg:", CONFIG_PATH)
  exp = SisuaExperimenter(ncpu=1)
  exp.run()

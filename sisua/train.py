from __future__ import absolute_import, division, print_function

import inspect
import os
import sys

import numpy as np
import tensorflow as tf

from odin.backend import interpolation
from odin.exp import Experimenter, ExperimentManager
from odin.utils.crypto import md5_checksum
from sisua.data import (CONFIG_PATH, DATA_DIR, EXP_DIR, get_dataset,
                        get_dataset_meta)
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


def random_variable(name: str,
                    cfg: dict,
                    dim: int = None,
                    posterior: str = None,
                    kwargs: dict = {},
                    required: bool = True):
  r""" Create RandomVariable from a named DictConfig

  name: the key name of the variable
  """
  if required and name not in cfg:
    raise ValueError("Cannot find variable with name='%s' in config: %s" %
                     (name, str(cfg)))
  if name not in cfg:
    return None
  rv = cfg[name]
  dim = rv.get('dim', dim)
  posterior = rv.get('posterior', posterior)
  kwargs = rv.get('kwargs', kwargs)
  if any(i is None for i in (dim, posterior, kwargs)):
    raise RuntimeError("Missing argument for random variable: "
                       "name=%s dim=%s posterior=%s kwargs=%s" %
                       (name, str(dim), str(posterior), str(kwargs)))
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
    gene, prot = get_dataset(cfg.dataset.name)
    split = float(cfg.get('split', 0.8))
    x_train, x_test = gene.split(split)
    if prot is not None:
      y_train, y_test = prot.split(split)
      x_train.assert_matching_cells(y_train)
      x_test.assert_matching_cells(y_test)
    else:
      y_train, y_test = None, None
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.rna_dim = x_train.shape[1]
    self.adt_dim = y_train.shape[1] if y_train is not None else None

  def on_create_model(self, cfg):
    network = NetworkConfig(**cfg.network)
    rv_latent = random_variable('latent', cfg)
    rv_rna = random_variable('rna', cfg, dim=self.rna_dim)
    rv_adt = random_variable('adt', cfg, dim=self.adt_dim, required=False)
    log_norm = cfg.get('log_norm', True)
    # ====== create model ====== #
    model_cls = get_model(cfg.model)
    args = inspect.getfullargspec(model_cls.__init__).args
    model_kwargs = dict(network=network, log_norm=log_norm)
    model_kwargs.update(elbo_config(cfg))
    if 'rna_dim' in args and 'adt_dim' in args:
      model_kwargs['rna_dim'] = rv_rna.dim
      model_kwargs['adt_dim'] = rv_adt.dim
    else:
      model_kwargs['outputs'] = [rv_rna, rv_adt
                                ] if rv_adt is not None else rv_rna
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
                                          ignores=['inputs'])
    self.model.fit(\
      inputs=self.x_train if self.y_train is None else [self.x_train, self.y_train],
      **kwargs)
    # save the best model after training
    save(model_path, self.model, max_to_keep=5)

  @property
  def args_help(self):
    all_models = [i.id for i in get_all_models()]
    all_datasets = list(get_dataset_meta().keys())
    return {'model': all_models, 'dataset.name': all_datasets}


# ===========================================================================
# Running the experiments
# ===========================================================================
if __name__ == "__main__":
  print("Path:")
  print(" - exp:", EXP_DIR)
  print(" - dat:", DATA_DIR)
  print(" - cfg:", CONFIG_PATH)
  exp = SisuaExperimenter(ncpu=1)
  exp.run()

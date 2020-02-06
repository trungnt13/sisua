from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from odin.exp import Experimenter, ExperimentManager
from odin.utils.crypto import md5_checksum
from sisua.data import get_dataset
from sisua.models import NetworkConfig, RandomVariable, get

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


class SisuaExperiment(Experimenter):

  def random_variable(self,
                      name,
                      cfg,
                      dim=None,
                      posterior=None,
                      kwargs={},
                      required=True):
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
    return RandomVariable(dim=int(dim),
                          posterior=posterior,
                          name=name,
                          **kwargs)

  def on_load_data(self, cfg):
    gene, prot = get_dataset(cfg.dataset_name)
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

  def on_load_model(self, cfg):
    network = NetworkConfig(**cfg.network)
    rv_latent = self.random_variable('latent', cfg)
    rv_rna = self.random_variable('rna', cfg, dim=self.rna_dim)
    rv_adt = self.random_variable('adt', cfg, dim=self.adt_dim, required=False)
    model = get(cfg.model)(
        outputs=[rv_rna, rv_adt] if rv_adt is not None else rv_rna,
        latents=rv_latent,
        network=network)
    self.model = model

  def on_train(self, cfg):
    kwargs = Experimenter.match_arguments(self.model.fit,
                                          cfg,
                                          ignores=['inputs'])
    self.model.fit(\
      inputs=self.x_train if self.y_train is None else [self.x_train, self.y_train],
      **kwargs)

  def on_clean(self, cfg):
    pass


exp = SisuaExperiment("/tmp/sisua",
                      "/data1/libs/sisua/configs/base.yaml",
                      ncpu=2)
exp.run(overrides=dict(model=['sisua', 'dca'],
                       dataset_name=['cortex', 'pbmc8kly'],
                       train=['adam', 'sgd']))
# manager = exp.manager

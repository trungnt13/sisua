from __future__ import absolute_import, division, print_function

import inspect
import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from odin.backend import interpolation
from odin.exp import Experimenter
from sisua.analysis import Posterior
from sisua.data import (CONFIG_PATH, DATA_DIR, EXP_DIR, OMIC, SingleCellOMIC,
                        get_dataset, get_dataset_meta)
from sisua.models import (NetworkConfig, RandomVariable, get_all_models,
                          get_model)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


# ===========================================================================
# Helpers
# ===========================================================================
def _from_config(cfg, fn, overrides={}):
  assert callable(fn)
  spec = inspect.getfullargspec(fn)
  kw = {
      k: v for k, v in cfg.items() if k in spec.args or spec.varkw is not None
  }
  overrides = {
      k: v
      for k, v in overrides.items()
      if k in spec.args or spec.varkw is not None
  }
  kw.update(overrides)
  return fn(**kw)


# ===========================================================================
# Main class
# ===========================================================================
class SisuaExperimenter(Experimenter):

  def __init__(self):
    super().__init__(save_path=EXP_DIR,
                     config_path=CONFIG_PATH,
                     exclude_keys=["train", "verbose"],
                     hash_length=5)
    print("Initialize SisuaExperimenter:")
    print(" - save   :", EXP_DIR)
    print(" - data   :", DATA_DIR)
    print(" - config :", CONFIG_PATH)

  def on_load_data(self, cfg):
    ds = cfg.dataset
    sco = get_dataset(ds.name)
    if cfg.verbose:
      print(sco)
    train, test = sco.split(train_percent=ds.train_percent)
    self.sco = sco
    self.train = train
    self.test = test

  def on_create_model(self, cfg, model_dir, md5):
    model = cfg.model
    cls = get_model(model.name)
    # parse networks
    encoder = _from_config(model.encoder, NetworkConfig)
    decoder = _from_config(model.decoder, NetworkConfig)
    # parse random variables
    omics = {o.name: self.sco.get_dim(o) for o in self.sco.omics}
    rv = {
        k:
        _from_config(v,
                     RandomVariable,
                     overrides=dict(
                         event_shape=omics[k] if k in omics else v.event_shape,
                         projection=True,
                         name=k))
        for k, v in cfg.variables.items()
        if k in omics or k in ('latents')
    }
    # create the model
    overrides = dict(outputs=rv['transcriptomic'],
                     latents=rv['latents'],
                     encoder=encoder,
                     decoder=decoder)
    # check if semi-supervised
    if 'labels' in inspect.getfullargspec(cls.__init__).args:
      # there might be case with no labels data available for semi-supervised
      # learning
      overrides['labels'] = [
          rv[o.name] for o in list(self.sco.omics)[1:] if o.name in rv
      ]
    if cfg.verbose:
      for i, j in overrides.items():
        print(f"{i}:\n\t{j}")
    # create the model
    self.model = _from_config(model, cls, overrides=overrides)
    self.model.load_weights(os.path.join(model_dir, 'model'),
                            verbose=cfg.verbose)
    # extract all necessary OMICs for training
    self.omics = [i.name for i in self.model.output_layers]
    if hasattr(self.model, 'labels'):
      for i in self.model.labels:
        self.omics.append(i.name)
    if cfg.verbose:
      print(self.model)
      print("Training OMICs:", self.omics)

  def on_train(self, cfg, output_dir, model_dir):
    self.model.set_metadata(self.sco)
    train, valid = self.train.split(0.9)
    train.corrupt(dropout_rate=cfg.dataset.dropout_rate,
                  retain_rate=cfg.dataset.retain_rate,
                  inplace=True)
    if cfg.verbose:
      print(train)
    train = train.create_dataset(self.omics,
                                 labels_percent=cfg.dataset.labels_percent,
                                 batch_size=cfg.dataset.batch_size,
                                 drop_remainder=True,
                                 shuffle=1000)
    valid = valid.create_dataset(self.omics,
                                 labels_percent=cfg.dataset.labels_percent,
                                 batch_size=cfg.dataset.batch_size,
                                 drop_remainder=True,
                                 shuffle=1000)
    if cfg.verbose:
      print(train)
    sample_shape = tuple(cfg.train.sample_shape)
    fn_save = partial(self.model.save_weights,
                      filepath=os.path.join(model_dir, 'model'))
    _from_config(cfg.train,
                 self.model.fit,
                 overrides=dict(log_tag=f"{cfg.model.name}-{cfg.dataset.name}",
                                train=train,
                                valid=valid,
                                sample_shape=sample_shape,
                                checkpoint=fn_save))

  def on_eval(self, cfg, output_dir):
    model = self.model
    dsname = model.dataset
    post = Posterior(scm=model, sco=self.test, batch_size=4)
    self.write_scores(table=f"imputation_{dsname}",
                      replace=True,
                      model=model.id,
                      **post.cal_imputation_scores())
    self.write_scores(table=f"llk_{dsname}",
                      replace=True,
                      model=model.id,
                      **post.cal_marginal_llk())
    if OMIC.proteomic in self.test.omics:
      self.write_scores(table=f"pearson_{dsname}",
                        replace=True,
                        model=model.id,
                        **post.cal_pearson())
      self.write_scores(table=f"spearman_{dsname}",
                        replace=True,
                        model=model.id,
                        **post.cal_spearman())
      self.write_scores(table=f"mi_{dsname}",
                        replace=True,
                        model=model.id,
                        **post.cal_mutual_information())
    # disentanglement scores
    if len(post._criticizers) > 0:
      crt = list(post._criticizers.values())[0]
      self.write_scores(
          table=f"disentanglement_{dsname}",
          replace=True,
          model=model.id,
          **crt.cal_clustering_scores(),
          **crt.cal_dci_scores(),
          **crt.cal_mutual_info_gap(),
          **crt.cal_total_correlation(),
          **crt.cal_separated_attr_predictability(),
          **crt.cal_relative_disentanglement_strength(),
          **crt.cal_relative_mutual_strength(),
          **crt.cal_betavae_score(),
          **crt.cal_factorvae_score(),
      )

  def on_plot(self, cfg, figure_dir):
    pass

  def on_compare(self, configs, models, save_path):
    pass


# ===========================================================================
# Running the experiments
# ===========================================================================
if __name__ == "__main__":
  exp = SisuaExperimenter()
  exp.run()

# Example:
# model.name=sisua,dca,vae,scvi,scale,misa,scalar,fvae,sfvae
# dataset.name=call,mpal,cortex,8kly,8klyall,eccly,ecclyall,vdj1,vdj1all,callx,mpalx,8kx,eccx,vdj1x,vdj4x
# -m -ncpu 4
# --reset

# model.name=sisua,dca,vae,scvi,scale,misa \
# dataset.name=8kly,8klyall,callx,mpalx,8kx,eccx,vdj1x,vdj4x,pbmcx,cbmcx \
# -m -ncpu 4 \
# --reset

# model.name=fvae,sisua,dca,vae,scvi \
# dataset.name=8kly,8klyall,mpalx,8kx,eccx,vdj4x,pbmcx \
# -m -ncpu 4 \
# --reset

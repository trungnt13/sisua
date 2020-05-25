from __future__ import absolute_import, division, print_function

import inspect
import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from odin.backend import interpolation
from odin.exp import Experimenter
from sisua.data import (CONFIG_PATH, DATA_DIR, EXP_DIR, OMIC, SingleCellOMIC,
                        get_dataset, get_dataset_meta)
from sisua.models import (NetworkConfig, RandomVariable, get_all_models,
                          get_model, load_model, save_model)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

# Example:
# model.name=sisua,dca,vae,scvi,scale,misa,scalar,fvae,sfvae
# dataset.name=cortex,8kly,8klyall,8klyx,eccly,ecclyall,ecclyx,8k,8kall
# -m -ncpu 5 --reset


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
    omics = {o.name: self.sco.get_omic_dim(o) for o in self.sco.omics}
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
      overrides['labels'] = [
          rv[o.name] for o in list(self.sco.omics)[1:] if o.name in rv
      ]
    if cfg.verbose:
      for i, j in overrides.items():
        print(f"{i}:\n\t{j}")
    self.model = _from_config(model, cls, overrides=overrides)
    self.model.dataset = cfg.dataset.name
    self.model.load_weights(os.path.join(model_dir, 'weights'),
                            verbose=cfg.verbose)
    self.omics = [i.name for i in self.sco.omics] \
      if self.model.is_semi_supervised else \
      ['transcriptomic']
    if cfg.verbose:
      print(self.model)
      print("Training OMICs:", self.omics)

  def on_train(self, cfg, output_dir, model_dir):
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
    _from_config(cfg.train,
                 self.model.fit,
                 overrides=dict(
                     train=train,
                     valid=valid,
                     sample_shape=sample_shape,
                     checkpoint=lambda: save_model(model_dir, self.model)))

  def on_eval(self, cfg, output_dir):
    model = self.model
    name = f"{model.id}-{model.dataset}"
    # very easy to OOM here, reduce batch size as much as possible
    mllk, llk = [], []
    prog = tqdm(self.test.create_dataset(self.omics,
                                         labels_percent=1.0,
                                         batch_size=2,
                                         drop_remainder=True),
                desc=f"[{name}] Marginal LLK")
    for data in prog:
      outputs = model.marginal_log_prob(**data, sample_shape=100)
      mllk.append(outputs[0])
      llk.append(outputs[1]['transcriptomic'])
    mllk = tf.reduce_mean(tf.concat(mllk, axis=0)).numpy()
    llk = tf.reduce_mean(tf.concat(llk, axis=0)).numpy()
    prog.clear()
    prog.close()
    # save the score
    self.write_scores('llk',
                      replace=True,
                      model=model.id,
                      dataset=model.dataset,
                      mllk=mllk,
                      llk=llk)

  def on_plot(self, cfg, output_dir):
    model = self.model
    name = f"{model.id}-{model.dataset}"
    # plot learning rate
    model.plot_learning_curves(
        os.path.join(output_dir, 'learning_curves.png'),
        summary_steps=[100, 10],
        title=f"{name}",
    )

  def on_compare(self, models, save_path):
    print(models)


# ===========================================================================
# Running the experiments
# ===========================================================================
if __name__ == "__main__":
  print("Path:")
  print(" - exp:", EXP_DIR)
  print(" - dat:", DATA_DIR)
  print(" - cfg:", CONFIG_PATH)
  exp = SisuaExperimenter()
  exp.run()

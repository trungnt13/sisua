from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from odin.bay import vi
from odin.exp import Experimenter
from odin.utils import get_formatted_datetime
from sisua.analysis import Posterior, ResultsSheet
from sisua.data import get_dataset
from sisua.train import SisuaExperimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
# model, cfg = exp.search(dict(model="sisua"))[0]

exp = SisuaExperimenter()
model, cfg = exp.random_model()
gene, prot = get_dataset(cfg.dataset.name)
split = float(cfg.dataset.get('split', 0.8))
x_train, x_test = gene.split(0.98)
if prot is not None:
  y_train, y_test = prot.split(split)

crt = vi.Criticizer(model)
with crt.generate_batch(x_test.X, y_test.X, discretizing=False):
  pass
exit()

pos = Posterior(scm=model,
                gene=x_test,
                protein=y_test,
                batch_size=16,
                n_mcmc=10)
pos.plot_latents_risk(seed=1)
pos.save_figures()
# pos.plot_latents_uncertainty_scatter()

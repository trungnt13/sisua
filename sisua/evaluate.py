from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from odin.exp import Experimenter
from odin.utils import get_formatted_datetime
from sisua.analysis import Posterior, ResultsSheet
from sisua.train import SisuaExperimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

exp = SisuaExperimenter()
model, cfg = exp.search(dict(model="sisua"))[0]

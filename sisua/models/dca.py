# Tensorflow re-implementaiton of: https://github.com/theislab/dca
# License: https://github.com/theislab/dca/blob/master/LICENSE.txt
# Modification by Trung Ngo 2019
from __future__ import absolute_import, division, print_function

import warnings

import tensorflow as tf

from sisua.models.single_cell_model import RandomVariable, SingleCellModel


class DeepCountAutoencoder(SingleCellModel):
  r""" Deep Count Autoencoder """

  def __init__(self,
               outputs,
               latents=RandomVariable(10, 'relu', True, name="Latents"),
               **kwargs):
    # force a deterministic latent space
    latents = tf.nest.flatten(latents)
    for z in latents:
      if not z.is_deterministic:
        warnings.warn(
            "DeepCountAutoencoder only support deterministic latents, "
            f"but given {z}, use default linear Dense layer for latents.")
        z.posterior = 'linear'
    super().__init__(outputs=outputs, latents=latents, **kwargs)

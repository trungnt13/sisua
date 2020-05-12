from __future__ import absolute_import, division, print_function

import warnings

import tensorflow as tf

from sisua.models.single_cell_model import RandomVariable, SingleCellModel
from sisua.models.vae import SISUA


class SCALE(SingleCellModel):
  r""" Tensorflow implementation of SCALE -
    "Single-Cell ATAC-seq analysis via Latent feature Extraction"

  Author: Lei Xiong - https://github.com/jsxlei
  License: https://github.com/jsxlei/SCALE/blob/master/LICENSE

   Reference:
    Xiong, L., Xu, K., Tian, K., et al., 2019. SCALE method for single-cell
      ATAC-seq analysis via latent feature extraction. Nature Communications.
      https://www.nature.com/articles/s41467-019-12630-7
  """

  def __init__(self,
               latents=RandomVariable(10, 'mixdiag', True, name="Latents"),
               latent_components=8,
               **kwargs):
    latents = tf.nest.flatten(latents)
    latent_components = int(latent_components)
    for z in latents:
      if z.posterior != 'mixdiag':
        warnings.warn("SCALE only allow 'mixdiag' posterior for latents, "
                      f"given: {z.posterior}")
      z.posterior = 'mixdiag'
      if 'n_components' not in z.kwargs:
        z.kwargs['n_components'] = latent_components
    super().__init__(latents=latents, analytic=False, **kwargs)


class SCALAR(SCALE, SISUA):
  r""" SCALE with semi-supervised extension -
    "Single-Cell ATAC-seq analysis via Latent and ADT Recombination"

  ADT: (antibody-derived tags)
  """

  def __init__(self, outputs, labels, **kwargs):
    super().__init__(outputs=outputs, labels=labels, **kwargs)

from __future__ import absolute_import, division, print_function

import warnings

import tensorflow as tf

from sisua.models.single_cell_model import RandomVariable, SingleCellModel
from sisua.models.vae import SISUA

__all__ = ['SCALE', 'SCALAR']


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
               outputs,
               latents=RandomVariable(10, 'mixgaus', True, name="Latents"),
               n_components=10,
               covariance='none',
               tie_mixtures=False,
               tie_loc=False,
               tie_scale=False,
               **kwargs):
    latents = tf.nest.flatten(latents)
    kw = dict(n_components=int(n_components),
              covariance=str(covariance),
              tie_mixtures=bool(tie_mixtures),
              tie_loc=bool(tie_loc),
              tie_scale=bool(tie_scale))
    for z in latents:
      if not z.is_mixture:
        warnings.warn("SCALE only allow mixture distribution for latents "
                      f" posterior, but given: {z.posterior}")
        z.posterior = 'mixgaus'
      for k, v in kw.items():
        if k not in z.kwargs:
          z.kwargs[k] = v
    super().__init__(outputs=outputs, latents=latents, analytic=False, **kwargs)


class SCALAR(SCALE, SISUA):
  r""" SCALE with semi-supervised extension -
    "Single-Cell ATAC-seq analysis via Latent and ADT Recombination"

  ADT: (antibody-derived tags)
  """

  def __init__(self, outputs, labels, **kwargs):
    super().__init__(outputs=outputs, labels=labels, **kwargs)

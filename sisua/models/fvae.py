from odin.bay.vi.autoencoder import factorVAE, SemifactorVAE
from sisua.models.single_cell_model import SingleCellModel

# note: this order of inheritant is the must

__all__ = ['FVAE', 'SemiFVAE']


class FVAE(factorVAE, SingleCellModel):

  def __init__(self, outputs, **kwargs):
    super().__init__(outputs=outputs, **kwargs)


class SemiFVAE(SemifactorVAE, SingleCellModel):

  def __init__(self, outputs, labels, **kwargs):
    super().__init__(outputs=outputs, labels=labels, **kwargs)

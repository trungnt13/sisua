from odin.bay.vi.autoencoder import FactorVAE, SemiFactorVAE
from sisua.models.single_cell_model import SingleCellModel

# note: this order of inheritant is the must

__all__ = ['FVAE', 'SemiFVAE']


class FVAE(FactorVAE, SingleCellModel):

  def __init__(self, outputs, **kwargs):
    super().__init__(outputs=outputs, **kwargs)


class SemiFVAE(SemiFactorVAE, SingleCellModel):

  def __init__(self, outputs, labels, **kwargs):
    super().__init__(outputs=outputs, labels=labels, **kwargs)

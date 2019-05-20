from __future__ import print_function, absolute_import, division

import os
import numpy as np

from scvi.dataset import GeneExpressionDataset
from sisua.data import get_dataset

class FacsDataset(GeneExpressionDataset):

  def __init__(self, n_proteins=7):
    assert n_proteins in (2, 5, 7), "Only support: 2, 5 or 7 protein FACS dataset"

    self.n_proteins = int(n_proteins)
    expression_data = self.download_and_preprocess()
    super().__init__(
        *GeneExpressionDataset.get_attributes_from_matrix(expression_data)
    )

  def preprocess(self):
    ds, gene_ds, prot_ds = get_dataset(
        dataset_name="facs_%d" % self.n_proteins, override=False)
    expression_data = gene_ds.X
    gene_symbols = gene_ds.X_col
    self.gene_symbols = gene_symbols
    self.cell_names = gene_ds.X_row

    self.adt_expression = prot_ds.X
    self.protein_markers = prot_ds.X_col

    assert np.all(gene_ds.X_row == prot_ds.X_row)

    return expression_data


class PbmcCiteseqDataset(GeneExpressionDataset):

  def __init__(self):
    expression_data = self.download_and_preprocess()
    super().__init__(
        *GeneExpressionDataset.get_attributes_from_matrix(expression_data)
    )

  def preprocess(self):
    ds, gene_ds, prot_ds = get_dataset(dataset_name="pbmc_citeseq",
                                       override=False)
    expression_data = gene_ds.X
    gene_symbols = gene_ds.X_col
    self.gene_symbols = gene_symbols
    self.cell_names = gene_ds.X_row

    self.adt_expression = prot_ds.X
    self.protein_markers = prot_ds.X_col

    assert np.all(gene_ds.X_row == prot_ds.X_row)

    return expression_data

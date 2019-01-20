import numpy as np

def imputation_score(original, imputed):
  assert original.shape == imputed.shape
  return np.median(np.abs(original - imputed))

from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict
from io import BytesIO

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.exp import Experimenter, ScoreBoard
from odin.utils import MPI, cache_disk, get_formatted_datetime
from sisua import get_dataset, get_dataset_meta, standardize_protein_name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)


@cache_disk
def get_corr(name):
  if 'all' == name[-3:] or 'hemato' in name:
    return
  try:
    ds = get_dataset(name, override=False, verbose=True)
  except Exception as e:
    print(f"Error loading dataset: '{name}'; error: {e}")
    return
  if 'proteomic' in ds.omics:
    var_names = ds.get_var_names("proteomic")
    print(ds.name, ":", var_names, ":", ds.X.shape[0])
    corr_data = ds.get_correlation('proteomic', 'proteomic')
    corr = {}
    corr_count = defaultdict(int)
    for i1, i2, p, s in corr_data:
      name1 = standardize_protein_name(var_names[i1])
      name2 = standardize_protein_name(var_names[i2])
      if name1 == name2:
        continue
      key = tuple(sorted([name1, name2]))
      corr[key] = min(p, s) if p < 0 and s < 0 else max(p, s)
      if p > 0.5 or s > 0.5:
        corr_count[key] += 1
      elif p < -0.5 or s < -0.5:
        corr_count[key] -= 1
    corr = sorted(corr.items(), key=lambda x: x[-1], reverse=True)
    corr_count = sorted(corr_count.items(), key=lambda x: x[-1], reverse=True)
    return ds.X.shape[0], var_names, corr, corr_count


# ===========================================================================
# Extract the minimum correlation
# ===========================================================================
dsname = list(get_dataset_meta().keys())
correlation = defaultdict(float)
count = defaultdict(float)
occurrence = defaultdict(int)
total_cell = 0

for results in MPI(jobs=dsname, func=get_corr, ncpu=4, batch=1):
  if results is None:
    continue
  n, var_names, corr, corr_count = results
  total_cell += n
  ## occurrence
  for v in var_names:
    occurrence[v] += 1
  ## correlation
  for i, j in corr:
    if i[0] == i[1]:
      continue
    correlation[i] += j
  ## count
  for i, j in corr_count:
    if i[0] == i[1]:
      continue
    count[i] += j

# sorting in descending order
occurrence = sorted(occurrence.items(), key=lambda x: x[-1], reverse=True)
correlation = sorted(correlation.items(), key=lambda x: x[-1], reverse=True)
count = sorted(count.items(), key=lambda x: x[-1], reverse=True)
occurrence = dict(occurrence)
# ===========================================================================
# Post-processing
# ===========================================================================
all_protein = ()
for i, j in correlation:
  all_protein += i
all_protein = set(all_protein)
print(list(occurrence.items()))
assert len(all_protein) == len(occurrence)

# ===========================================================================
# Show top and bottom
# ===========================================================================
n = 100
print("Total cells  :", total_cell)
print("Total pairs  :", len(correlation))
print("Total protein:", len(all_protein))
#
print("=== Top ===")
print(correlation[:n])
print([i[0] for i in correlation[:n]])
print("=== Bottom ===")
print(correlation[-n:][::-1])
print([i[0] for i in correlation[-n:]][::-1])
#
print()
print("=== Top ===")
print(count[:n])
print([i[0] for i in count[:n]])
print("=== Bottom ===")
print(count[-n:][::-1])
print([i[0] for i in count[-n:]][::-1])

# ===========================================================================
# Top and bottom for each protein
# ===========================================================================
protein_positive = defaultdict(lambda: defaultdict(float))
protein_negative = defaultdict(lambda: defaultdict(float))
for (name1, name2), corr in correlation:
  if name1 == name2:
    continue
  d = protein_positive if corr > 0 else protein_negative
  d[name1][name2] = corr
  d[name2][name2] = corr
###
protein_positive = {
    k: sorted(v.items(), key=lambda x: x[-1], reverse=True)
    for k, v in protein_positive.items()
}
print("=== Positive ===")
for i, j in sorted(protein_positive.items(),
                   key=lambda x: occurrence.get(x[0], 0),
                   reverse=True):
  print(i, [k[0] for k in j[:5]])
###
protein_negative = {
    k: sorted(v.items(), key=lambda x: x[-1], reverse=False)
    for k, v in protein_negative.items()
}
print("=== Negative ===")
for i, j in sorted(protein_negative.items(),
                   key=lambda x: occurrence.get(x[0], 0),
                   reverse=True):
  print(i, [k[0] for k in j[:5]])

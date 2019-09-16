from __future__ import absolute_import, division, print_function

import os

import numpy as np
from six import string_types

from odin.utils import ctext


# ===========================================================================
# For saving data
# ===========================================================================
def save_data(outpath, header, row, data):
  """ Simple shortcut

  If `feather-format` is available, save data to `.feather` file
  otherwise `.csv`
  """
  try:
    import feather
    return save_data_to_R(outpath, header, row, data)
  except ImportError as e:
    return save_data_to_csv(outpath, header, row, data)


def save_data_to_csv(outpath, header, row, data):
  if data is None:
    return
  if '.csv' not in outpath:
    outpath += '.csv'

  assert len(row) == len(data), "Data length mismatch!"
  with open(outpath, 'w') as f:
    # preprocessing the head
    if isinstance(header, (tuple, list, np.ndarray, set)):
      header = ",".join([
          str(val) if isinstance(val, (string_types, np.string_)) else "D%d" % i
          for i, val in enumerate(header)
      ])
    f.write('Cell,' + header + '\n')
    # write the data
    for i, j in zip(row, data):
      f.write(i + ',')
      f.write(','.join(['%g' % x for x in j]))
      f.write('\n')


def save_data_to_R(outpath, header, row, data):
  if data is None:
    return
  if '.feather' not in outpath:
    outpath += '.feather'

  import pandas as pd
  try:
    import feather
  except ImportError as e:
    raise RuntimeError(
        "Cannot export to R, require python package 'feather-format'")

  row = np.array(row)
  if isinstance(header, string_types):
    header = header.split(',')
  header = np.array(header)
  df = pd.DataFrame(data=data, index=row, columns=header, dtype=data.dtype)

  feather.write_dataframe(df, outpath)

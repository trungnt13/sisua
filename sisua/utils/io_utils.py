from __future__ import print_function, division, absolute_import

import os
import numpy as np
from six import string_types
from odin.utils import ctext

def save_data_to_csv(outpath, header, row, data, print_log=False):
  if data is not None:
    if print_log:
      print("Saving '%s' to '%s' ..." % (ctext(data.shape, 'cyan'),
                                         ctext(outpath, 'yellow')))
    with open(outpath, 'w') as f:
      # preprocessing the head
      if isinstance(header, (tuple, list, np.ndarray, set)):
        header = ",".join([str(val)
                           if isinstance(val, (string_types, np.string_)) else
                           "D%d" % i
                           for i, val in enumerate(header)])
      f.write('Cell,' + header + '\n')
      # write the data
      for i, j in zip(row, data):
        f.write(i + ',')
        f.write(','.join(['%g' % x for x in j]))
        f.write('\n')

def load_npz_sorted(path):
  """ This always return a list of loaded results """
  return [i[1] for i in
          sorted(np.load(path).items(),
                 key=lambda x: int(x[0].split('_')[-1]))]

def check_and_load_npz(path, title, print_log=False):
  """ This load all: train, valid, test, test_dropout
  and return if the the data is loaded or file not found

  Return
  ------
  (train, test), is_loaded
  """
  is_loaded = False
  if os.path.exists(path) and os.path.isfile(path):
    if print_log:
      print("Loading [%s] data at: %s" % (ctext(title, 'lightyellow'), path))
    data = np.load(path)
    (train, test) = data['train'], data['test']
    is_loaded = True
  else:
    (train, test) = None, None
  return (train, test), is_loaded

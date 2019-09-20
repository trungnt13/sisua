from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sisua.data import get_dataset, get_dataset_meta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

for i, fn in get_dataset_meta().items():
  try:
    ds = fn()
    print("Loaded:", i)
  except Exception as e:
    print("Error:", i, e)

print()

df = []
fmt = '%s'
for i, name in enumerate(sorted(get_dataset_meta())):
  print(name)
  try:
    x, y = get_dataset(name)
    df.append([
        name, fmt % str(x).replace('\n', '<br>'),
        fmt % str(y).replace('\n', '<br>')
    ])
  except Exception as e:
    print("Error:", e)

pd.options.display.max_colwidth = 100000
df = pd.DataFrame(df, columns=['Name', 'Gene', 'Protein'])
html = df.to_html(justify='center', escape=False)
html = html.replace('<td>', '<td style="white-space:pre">')
with open('/tmp/tmp.html', 'w') as f:
  f.write(html)

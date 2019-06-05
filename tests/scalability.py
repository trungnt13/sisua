from __future__ import print_function, division, absolute_import

import time
import numpy as np
import pandas as pd
from collections import defaultdict

from odin.utils import ctext, ArgController

from sisua.inference import Inference, InferenceSCVI, InferenceDCA

args = ArgController(
).add('-path', "Save path csv file", '/tmp/tmp.csv'
).add('--sisua', "running SISUA first then scVAE", False
).add('--test', "evaluate the test time", False
).parse()

# ===========================================================================
# Configurations
# ===========================================================================
SEED = 87654321
n_cells = [200, 500, 1000, 2000, 5000, 10000, 40000, 100000]
n_genes = [500]
n_proteins = [10]

n_epoch = 100
batch_size = 128
n_trials = 1

np.random.seed(SEED)

save_path = args.path
print("Save path:", ctext(save_path, 'lightyellow'))

# ===========================================================================
# Helper
# ===========================================================================
def random_system():
  all_system = []
  np.random.permutation(2)
  for i in [1, 0] if bool(args.sisua) else [0, 1]:
    if i == 0:
      model = Inference(ngene, model='vae')
      name = "scVAE"
    elif i == 1:
      model = Inference(ngene, nprotein, model='movae')
      name = "SISUA"
    # elif i == 2:
    #   model = InferenceSCVI(ngene)
    #   name = "scVI"
    # elif i == 3:
    #   model = InferenceDCA(ngene)
    #   name = "DCA"
    all_system.append((name, model))
  return all_system

# ===========================================================================
# Exp
# ===========================================================================
results = []
for ncell in n_cells:
  for ngene in n_genes:
    for nprotein in n_proteins:
      X = np.random.randint(0, 100, size=(ncell, ngene))
      y = np.random.randint(0, 10, size=(ncell, nprotein))

      for trial in range(n_trials):
        for name, model in random_system():

          if bool(args.test):
            model.fit(X=X, y=y,
                      n_epoch=1, batch_size=batch_size,
                      detail_logging=False)
            start_time = time.time()
            model.predict_Z(X)
            model.predict_V(X)
            duration = time.time() - start_time
          else:
            start_time = time.time()
            model.fit(X=X, y=y,
                      n_epoch=n_epoch, batch_size=batch_size,
                      detail_logging=False)
          duration = time.time() - start_time
          print(ctext(name, 'cyan'), ncell, ngene, nprotein, trial, '%.2f(s)' % duration)

          results.append(dict(
              name=name,
              time=duration,
              ncell=ncell,
              ngene=ngene,
              nprotein=nprotein
          ))

df = pd.DataFrame(results)
print(df)
df.to_csv(save_path)

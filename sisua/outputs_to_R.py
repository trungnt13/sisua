from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
import shutil
import pickle
import time

import numpy as np

from odin import backend as K
from odin.utils import ctext, Progbar, ArgController

from sisua import set_verbose
from sisua.data import EXP_DIR, get_dataset
from sisua.utils import (save_data,
                         load_npz_sorted, check_and_load_npz,
                         filtering_experiment_path)
from sisua.label_threshold import GMMThresholding

# ===========================================================================
# Processing
# example: python sisua/outputs_to_R.py /data1/czi_results -ds facs_2 --override
# ===========================================================================
def main():
  set_verbose(False)

  args = ArgController(
  ).add('path', 'output path for storing results'
  ).add('--override', 'automatically override all exist path at the output dir', False
  ).parse()
  # ******************** pre-processing arguments ******************** #
  override = bool(args.override)
  OUTPUT_PATH = args.path

  if os.path.exists(OUTPUT_PATH):
    if not override:
      raise RuntimeError("Cannot override")
    else:
      shutil.rmtree(OUTPUT_PATH)

  if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
  assert os.path.isdir(OUTPUT_PATH)
  # ******************** converting outputs ******************** #
  for corruption in os.listdir(EXP_DIR):
    if not ('uniform' in corruption or 'binomial'in corruption):
      continue
    print(ctext(corruption, 'lightyellow'))
    corr_path = os.path.join(EXP_DIR, corruption)

    output_path = os.path.join(OUTPUT_PATH, corruption)
    if not os.path.exists(output_path):
      os.mkdir(output_path)

    for ds_name in os.listdir(corr_path):
      print(" ", ctext(ds_name, "lightcyan"))
      ds_path = os.path.join(corr_path, ds_name)
      ds_output_path = os.path.join(output_path, ds_name)
      if not os.path.exists(ds_output_path):
        os.mkdir(ds_output_path)

      ds, gene_ds, prot_ds = get_dataset(ds_name, override=False)

      # ====== save gene data ====== #
      X_train_raw = gene_ds.get_data('train')
      X_test_raw = gene_ds.get_data('test')

      X_col = gene_ds.col_name
      x_header = ",".join(X_col)
      row_train, row_test = gene_ds.row_name

      save_data(outpath=os.path.join(ds_output_path, "X_train_raw"),
                header=x_header, row=row_train, data=X_train_raw)
      save_data(outpath=os.path.join(ds_output_path, "X_test_raw"),
                header=x_header, row=row_test, data=X_test_raw)

      # ====== save protein data ====== #
      y_header = prot_ds.col_name
      y_train_raw = prot_ds.get_data('train')
      y_test_raw = prot_ds.get_data('test')
      gmm = GMMThresholding()
      gmm.fit(y_train_raw)

      save_data(outpath=os.path.join(ds_output_path, "y_train_raw"),
                header=y_header, row=row_train, data=y_train_raw)
      save_data(outpath=os.path.join(ds_output_path, "y_train_bin"),
                header=y_header, row=row_train, data=gmm.predict(y_train_raw))
      save_data(outpath=os.path.join(ds_output_path, "y_train_prob"),
                header=y_header, row=row_train, data=gmm.predict_proba(y_train_raw))

      save_data(outpath=os.path.join(ds_output_path, "y_test_raw"),
                header=y_header, row=row_test, data=y_test_raw)
      save_data(outpath=os.path.join(ds_output_path, "y_test_bin"),
                header=y_header, row=row_test, data=gmm.predict(y_test_raw))
      save_data(outpath=os.path.join(ds_output_path, "y_test_prob"),
                header=y_header, row=row_test, data=gmm.predict_proba(y_test_raw))

      for sys_name in os.listdir(ds_path):
        if 'vae' not in sys_name:
          continue
        # ====== loading model ====== #
        sys_path = os.path.join(ds_path, sys_name)
        model_path = os.path.join(sys_path, 'model.pkl')
        if not os.path.exists(model_path):
          continue
        with open(model_path, 'rb') as f:
          infer = pickle.load(f)
        # ====== get trained data ====== #
        ximpu = infer.config['ximpu']
        cdist = infer.config['cdist']
        X_train_imp = gene_ds.get_data(data_type='train',
          dropout=ximpu, distribution=cdist)
        X_test_imp = gene_ds.get_data(data_type='test',
          dropout=ximpu, distribution=cdist)
        # ====== latent space ====== #
        start_time = time.time()
        Z_train = infer.predict_Z(X_train_imp)
        Z_test = infer.predict_Z(X_test_imp)
        W_train = infer.predict_W(X_train_imp)
        W_test = infer.predict_W(X_test_imp)
        V_train = infer.predict_V(X_train_imp)
        V_test = infer.predict_V(X_test_imp)
        # ====== saving the output ====== #
        model_outpath = os.path.join(ds_output_path, sys_name)
        if not os.path.exists(model_outpath):
          os.mkdir(model_outpath)
        # saving gene-expression
        save_data(outpath=os.path.join(model_outpath, "W_train"),
                  header=x_header, row=row_train, data=W_train[0])
        save_data(outpath=os.path.join(model_outpath, "V_train"),
                  header=x_header, row=row_train, data=V_train[0])
        save_data(outpath=os.path.join(model_outpath, "W_test"),
                  header=x_header, row=row_test, data=W_test[0])
        save_data(outpath=os.path.join(model_outpath, "V_test"),
                  header=x_header, row=row_test, data=V_test[0])
        # saving latent space
        z_header = ",".join(["D%d" % i for i in range(Z_test.shape[1])])
        save_data(outpath=os.path.join(model_outpath, "Z_train"),
                  header=z_header, row=row_train, data=Z_train)
        save_data(outpath=os.path.join(model_outpath, "Z_test"),
                  header=z_header, row=row_test, data=Z_test)
        print("   Saving %s in %d(s)" % (ctext(sys_name, 'cyan'), time.time() - start_time))

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  main()

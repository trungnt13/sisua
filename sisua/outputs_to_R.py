from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
import shutil
import pickle

import numpy as np

from odin import backend as K
from odin.utils import ctext, Progbar, ArgController

from sisua import set_verbose
from sisua.data import EXP_DIR, get_dataset
from sisua.utils import (save_data_to_csv, save_data_to_R,
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
  ).add('path', 'output path for storing Rdata results'
  ).add('-ds', 'name of the dataset, if not given, all datasets are processed', ''
  ).add('-incl', 'keywords for including', ''
  ).add('-excl', 'keywords for excluding', ''
  ).add('--csv', 'save data to .csv format instead of Rdata', False
  ).add('--override', 'automatically override all exist path at the output dir', False
  ).parse()
  # ******************** pre-processing arguments ******************** #
  OUTPUT_PATH = args.path
  if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
  assert os.path.isdir(OUTPUT_PATH)

  if len(args.ds) == 0:
    dataset = os.listdir(EXP_DIR)
  else:
    dataset = [str(args.ds)]
  assert len(dataset) > 0, "No results found at path: %s" % EXP_DIR

  override = bool(args.override)
  f_save = save_data_to_csv if bool(args.csv) else save_data_to_R
  # ******************** converting outputs ******************** #
  for ds_name in dataset:
    all_exp, ds_name, gene_ds, prot_ds = filtering_experiment_path(
        ds_name_or_path=ds_name,
        incl_keywords=args.incl, excl_keywords=args.excl,
        return_dataset=True)
    if len(all_exp) == 0:
      continue
    print("Processing dataset:", ctext(ds_name, 'lightyellow'))

    X_train_raw = gene_ds.get_data('train')
    X_train_log = K.log_norm(X_train_raw, axis=1)

    X_test_raw = gene_ds.get_data('test')
    X_test_log = K.log_norm(X_test_raw, axis=1)

    X_col = gene_ds.col_name
    x_header = ",".join(X_col)
    row_train, row_test = gene_ds.row_name
    # ====== delete old files ====== #
    output_path = os.path.join(OUTPUT_PATH,
                               os.path.basename(ds_name))
    if os.path.exists(output_path):
      if not override and len(os.listdir(output_path)) > 0:
        raise RuntimeError(
            "path '%s' exists, but not allowed overriding" % output_path)
      shutil.rmtree(output_path)
    os.mkdir(output_path)
    print(" Output path:", ctext(output_path, 'cyan'))
    # ====== save basic data ====== #
    f_save(outpath=os.path.join(output_path, "X_train_raw"),
           header=x_header, row=row_train, data=X_train_raw)
    f_save(outpath=os.path.join(output_path, "X_train_log"),
           header=x_header, row=row_train, data=X_train_log)

    f_save(outpath=os.path.join(output_path, "X_test_raw"),
           header=x_header, row=row_test, data=X_test_raw)
    f_save(outpath=os.path.join(output_path, "X_test_log"),
           header=x_header, row=row_test, data=X_test_log)
    # ====== saving protein markers ====== #
    y_header = prot_ds.col_name
    y_train_raw = prot_ds.get_data('train')
    y_test_raw = prot_ds.get_data('test')

    gmm = GMMThresholding()
    gmm.fit(y_train_raw)

    f_save(outpath=os.path.join(output_path, "y_train_raw"),
           header=y_header, row=row_train, data=y_train_raw)
    f_save(outpath=os.path.join(output_path, "y_train_bin"),
           header=y_header, row=row_train, data=gmm.predict(y_train_raw))
    f_save(outpath=os.path.join(output_path, "y_train_prob"),
           header=y_header, row=row_train, data=gmm.predict_proba(y_train_raw))

    f_save(outpath=os.path.join(output_path, "y_test_raw"),
           header=y_header, row=row_test, data=y_test_raw)
    f_save(outpath=os.path.join(output_path, "y_test_bin"),
           header=y_header, row=row_test, data=gmm.predict(y_test_raw))
    f_save(outpath=os.path.join(output_path, "y_test_prob"),
           header=y_header, row=row_test, data=gmm.predict_proba(y_test_raw))
    # ====== filtering ====== #
    ds_path = os.path.join(EXP_DIR, ds_name)
    for name in os.listdir(ds_path):
      path = os.path.join(ds_path, name)
      model_path = os.path.join(path, 'model.pkl')
      # model is not fitted
      if not os.path.isdir(path) or not os.path.exists(model_path):
        continue
      print(" * ", ctext(os.path.basename(name), 'yellow'))
      # load the model
      with open(model_path, 'rb') as f:
        infer = pickle.load(f)
      with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

      ximpu = config['ximpu']
      yimpu = config['yimpu']
      n_samples = config['nsample_test']
      # ====== data ====== #
      X_train_imp = gene_ds.get_data(data_type='train', dropout=ximpu)
      X_test_imp = gene_ds.get_data(data_type='test', dropout=ximpu)

      X_train_imp_log = K.log_norm(X_train_imp, axis=1)
      X_test_imp_log = K.log_norm(X_test_imp, axis=1)

      y_train_imp = prot_ds.get_data(data_type='train', dropout=yimpu)
      y_test_imp = prot_ds.get_data(data_type='test', dropout=yimpu)
      # ====== latent space ====== #
      Z_train = infer.predict_Z(X_train_imp, y_train_imp,
                                n_mcmc_samples=n_samples)
      Z_test = infer.predict_Z(X_test_imp, y_test_imp,
                               n_mcmc_samples=n_samples)
      print("  - Predict latent space")
      # ====== post processing reconstruction values ====== #
      W_train = infer.predict_W(X_train_imp, y_train_imp,
                                n_mcmc_samples=n_samples)
      W_test = infer.predict_W(X_test_imp, y_test_imp,
                               n_mcmc_samples=n_samples)
      print("  - Predict reconstruction")
      V_train = infer.predict_V(X_train_imp, y_train_imp,
                                n_mcmc_samples=n_samples)
      V_test = infer.predict_V(X_test_imp, y_test_imp,
                               n_mcmc_samples=n_samples)
      print("  - Predict imputation")
      # ====== saving the output ====== #
      model_outpath = os.path.join(output_path, name)
      if not os.path.exists(model_outpath):
        os.mkdir(model_outpath)

      # saving imputed values
      f_save(outpath=os.path.join(model_outpath, "X_train_imp_raw"),
             header=x_header, row=row_train, data=X_train_imp)
      f_save(outpath=os.path.join(model_outpath, "X_test_imp_raw"),
             header=x_header, row=row_test, data=X_test_imp)
      f_save(outpath=os.path.join(model_outpath, "X_train_imp_log"),
             header=x_header, row=row_train, data=X_train_imp_log)
      f_save(outpath=os.path.join(model_outpath, "X_test_imp_log"),
             header=x_header, row=row_test, data=X_test_imp_log)
      print("  - Saved imputation values")

      # saving gene-expression
      f_save(outpath=os.path.join(model_outpath, "W_train"),
             header=x_header, row=row_train, data=W_train[0])
      f_save(outpath=os.path.join(model_outpath, "V_train"),
             header=x_header, row=row_train, data=V_train[0])

      f_save(outpath=os.path.join(model_outpath, "W_test"),
             header=x_header, row=row_test, data=W_test[0])
      f_save(outpath=os.path.join(model_outpath, "V_test"),
             header=x_header, row=row_test, data=V_test[0])

      # saving latent space
      z_header = ",".join(["D%d" % i for i in range(Z_test.shape[1])])
      f_save(outpath=os.path.join(model_outpath, "Z_train.csv"),
             header=z_header, row=row_train, data=Z_train)
      f_save(outpath=os.path.join(model_outpath, "Z_test.csv"),
             header=z_header, row=row_test, data=Z_test)
      print("  - Saved values to:", model_outpath)

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  main()

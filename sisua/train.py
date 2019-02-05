from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from odin.utils import catch_warnings_error

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'
import shutil
import pickle
import multiprocessing

import numpy as np
import tensorflow as tf

from odin.stats import describe, sparsity_percentage
from odin import nnet as N, backend as K, training as T, visual as V
from odin.ml import LogisticRegression, evaluate, fast_pca
from odin.utils import (unique_labels, ctext, auto_logging, batching,
                        UnitTimer, ArgController, get_script_path,
                        mpi, stdio, one_hot, Progbar, async_mpi,
                        md5_checksum)

from sisua import set_verbose
from sisua.data import (get_dataset, EXP_DIR, UNIVERSAL_RANDOM_SEED,
                        DROPOUT_TEST)
from sisua.analysis.latent_benchmarks import plot_latents
from sisua.utils import (plot_evaluate_classifier,
                         plot_evaluate_reconstruction, plot_evaluate_regressor,
                         fast_scatter, show_image, plot_monitoring_epoch,
                         LearningCurves)
from sisua.inference import Inference

def main():
  args = ArgController(
  ).add('-model',
        'name for model, specified in models_.py',
        'unsupervised'
  ).add('-ds',
        'name of the dataset: mnist, pbmc citeseq, pbmc 10xPP, cbmc, etc.',
        'facs_7'
  ).add('-xnorm',
        'normalization method for input data: log, bin, raw, prob',
        'log',
        ('log', 'bin', 'raw', 'prob')
  ).add('-ximpu',
        'The percents for dropout entries to test imputation on X (gene count)',
        0,
  ).add('-tnorm',
        'normalization method for target reconstruction data',
        'raw',
        ('log', 'bin', 'raw', 'prob')
  ).add('-ynorm',
        'processing method for labels: log, bin, raw, prob',
        'prob',
        ('log', 'bin', 'raw', 'prob')
  ).add('-yimpu',
        'The percents for dropout entries to test imputation on Y (protein marker)',
        0,
  ).add('-xclip', 'maximum value if training data to be clipped to', 0
  ).add('-yclip', 'maximum value if labels count to be clipped to', 0
  # ====== for VAE ====== #
  ).add('-xdist', 'input(X) distribution', 'zinb'
  ).add('-ydist', 'label(y) distribution', 'bernoulli'
  ).add('-zdist', 'latent distribution', 'normal'
  ).add('-cdist', 'Distribution for the corruption method: uniform or binomial', 'uniform'
  # ====== for training ====== #
  ).add('-lr', 'Set learning rate', 1e-4
  ).add('-epoch', 'number of training epoch', 120
  ).add('-batch', 'batch size for training', 32
  # ====== semi-supervised ====== #
  ).add('-nsample-train', 'number of MCMC sample during training', 1
  ).add('-nsample-test', 'number of MCMC sample during testing', 250
  ).add('-ps', 'percentage of training data for supervised task', 0.8
  ).add('-hdim', 'number of dimension for hidden layer', 256
  ).add('-zdim', 'number of dimension for latent space', 64
  ).add('-nlayer', 'number of dense layers', 2
  # ====== dropout ====== #
  ).add('-xdrop', 'dropout level for input X', 0.3
  ).add('-edrop', 'dropout level for ENCODER output before the latent', 0
  ).add('-ddrop', 'dropout level for DECODER output before reconstruction', 0
  ).add('-zdrop', 'dropout level on latent space', 0
  # ====== data ====== #
  ).add('--no-batchnorm', 'turn-off batch normalization', False
  ).add('--count-sum', 'introducing count-sum to the latent space', False
  ).add('--constraint', 'introducing count-sum to the reconstruction distribution', False
  ).add('--monitor', 'enable monitoring every epoch', False
  ).add('--no-iw', 'turn off: important weight sampling', False
  ).add('--no-analytic', 'turn off: analytic KL for variational inference', False
  ).add('--no-save', 'turn off saving the latent and reconstruction matrices into csv', False
  ).add('--override', 'override previous experiments', False
  ).parse()
  args['batchnorm'] = not args['no_batchnorm']
  args['iw'] = not args['no_iw']
  args['analytic'] = not args['no_analytic']

  # ====== validate the arguments ====== #
  assert (0 < args.nlayer < 100 and
          0 < args.hdim < 1000 and
          0 < args.zdim < 1000)
  assert (0 <= args.ximpu < 1 and
          0 <= args.yimpu < 1)
  assert (0 <= args.xdrop < 1 and
          0 <= args.edrop < 1 and
          0 <= args.zdrop < 1 and
          0 <= args.ddrop < 1)
  assert (args.xclip >= 0 and
          args.yclip >= 0)
  assert 0.0 < args.ps <= 1.0, \
      "`ps` value must be > 0 and <= 1, but given: %f" % float(args.ps)
  # ===========================================================================
  # Loading data
  # ===========================================================================
  set_verbose(True)
  (ds, gene_ds, prot_ds) = get_dataset(args.ds, xclip=args.xclip, yclip=args.yclip,
                                       override=False)
  # ====== data for training and testing ====== #
  X_train = gene_ds.get_data(data_type='train',
    dropout=args['ximpu'], distribution=args.cdist)
  y_train = prot_ds.get_data(data_type='train',
    dropout=args['yimpu'], distribution=args.cdist)
  X_test = gene_ds.get_data(data_type='test',
    dropout=args['ximpu'], distribution=args.cdist)
  y_test = prot_ds.get_data(data_type='test',
    dropout=args['yimpu'], distribution=args.cdist)
  # ===========================================================================
  # model identify
  # ===========================================================================
  # [0] model name
  # [1] feat name
  # [2] percentage of supervised data
  # [3] latent dimension
  # [4] use mouse genes or not
  MODEL_NAME = str(args.model).strip().lower()
  MODEL_ID = '_'.join([
      MODEL_NAME,
      'X%s%d%s' % (args.xnorm, int(max(args.xclip, 0)), args.xdist),
      'Y%s%d%s' % (args.ynorm, int(max(args.yclip, 0)), args.ydist),
      'T%s' % (args.tnorm),
      'Z' + str(args.zdist),
      'I%.2d%.2d%s' % (args.ximpu, args.yimpu, args.cdist),
      'mcTrn%dTst%d' % (int(args.nsample_train), int(args.nsample_test)),
      'spvs%.3d' % (args.ps * 100),
      'net%.2d%.3d%.3d' % (args.nlayer, args.hdim, args.zdim),
      'drop%.2d%.2d%.2d%.2d' % (args.xdrop * 100,
                                args.edrop * 100,
                                args.zdrop * 100,
                                args.ddrop * 100),
      'alytcT' if bool(args.analytic) else 'alytcF',
      'iwT' if bool(args.iw) else 'iwF',
      'bnormT' if bool(args.batchnorm) else 'bnormF',
      'cntsmT' if bool(args.count_sum) else 'cntsmF',
      'cstrnT' if bool(args.constraint) else 'cstrnF',
  ])
  BASE_DIR = os.path.join(EXP_DIR, ds.name)
  if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

  MODEL_DIR = os.path.join(BASE_DIR, MODEL_ID)
  if bool(args.override) and os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  # ====== save the args ====== #
  with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump(args, f)

  # ====== print some log ====== #
  stdio(os.path.join(MODEL_DIR, 'init.log'))
  print(ctext("Model ID:", 'cyan'), ctext(MODEL_ID, 'lightred'))
  print(ctext("Base dir:", 'cyan'), ctext(BASE_DIR, 'lightred'))
  # ===========================================================================
  # Training the network
  # ===========================================================================
  stdio(os.path.join(MODEL_DIR, 'train.log'))
  infer = Inference(model_name=MODEL_NAME,
                    model_config=args,
                    cellsize_normalize_factor=gene_ds.cell_median)
  infer.fit(X=X_train, y=y_train,
            supervised_percent=args['ps'], validation_percent=0.1,
            n_mcmc_samples=args['nsample_train'],
            batch_size=args['batch'], n_epoch=args['epoch'],
            learning_rate=args['lr'],
            monitoring=False, fig_dir=MODEL_DIR,
            detail_logging=True)
  # ====== save the trained model ====== #
  with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(infer, f)
  # ===========================================================================
  # Evaluation and make prediction
  # ===========================================================================
  stdio(os.path.join(MODEL_DIR, 'score.log'))
  for name, x, y in zip(("Train", "Test"),
                        (X_train, X_test),
                        (y_train, y_test)):
    scores = infer.score(x, y, n_mcmc_samples=args['nsample_test'])
    print("======== %s ========" % name)
    for k, v in scores.items():
      print("  ", '%-32s' % k.split(":")[0].split('/')[-1], v)
# ===========================================================================
# Calling the main
# ===========================================================================
if __name__ == '__main__':
  main()

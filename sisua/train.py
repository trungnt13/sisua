from __future__ import print_function, division, absolute_import
import re
import numpy as np
from itertools import product

def get_arguments():
  """ Multiple values are provided by ',' """
  from odin.utils import ArgController
  args = ArgController(
  ).add('-model',
        'name for model, specified in models_.py; with the exception is "scvi" and "dca"',
        'movae'
  ).add('-ds',
        'name of the dataset: mnist, pbmc citeseq, pbmc 10xPP, cbmc, etc.',
        'pbmc8k_ly'
  ).add('-dispersion',
        'dispersion type: per "gene" or per "gene-cell"',
        'gene-cell',
        ('gene', 'gene-cell')
  ).add('-xnorm',
        'normalization method for input data: log, bin, raw, prob',
        'log',
        ('log', 'bin', 'raw', 'prob')
  ).add('-ximpu',
        'The percents for dropout entries to test imputation on X (gene count)',
        0.25,
  ).add('-tnorm',
        'normalization method for target reconstruction data',
        'raw',
        ('log', 'bin', 'raw', 'prob')
  ).add('-ynorm',
        'processing method for labels: log, bin, raw, prob',
        'prob',
        ('log', 'bin', 'raw', 'prob')
  ).add('-xclip', 'maximum value if training data to be clipped to', 0
  ).add('-yclip', 'maximum value if labels count to be clipped to', 0
  # ====== for VAE ====== #
  ).add('-xdist', 'input(X) distribution', 'zinb'
  ).add('-ydist', 'label(y) distribution', 'bernoulli'
  ).add('-zdist', 'latent distribution', 'normal'
  ).add('-cdist',
        'Distribution for the corruption method: uniform or binomial',
        'binomial'
  # ====== for training ====== #
  ).add('-lr', 'Set learning rate', 1e-4
  ).add('-epoch', 'number of training epoch', 180
  ).add('-batch', 'batch size for training', 128
  # ====== semi-supervised ====== #
  ).add('-nsample', 'number of MCMC sample during training', 1
  ).add('-ps', 'percentage of training data for supervised task', 0.8
  ).add('-hdim', 'number of dimension for hidden layer', 128
  ).add('-zdim', 'number of dimension for latent space', 32
  ).add('-nlayer', 'number of dense layers', 2
  # ====== dropout ====== #
  ).add('-xdrop', 'dropout level for input X', 0.3
  ).add('-edrop', 'dropout level for ENCODER output before the latent', 0
  ).add('-ddrop', 'dropout level for DECODER output before reconstruction', 0
  ).add('-zdrop', 'dropout level on latent space', 0
  # ====== kl-divergence ====== #
  ).add('-kl-weight',
        'weight for KL(q|p) divergence during training', 1
  ).add('-warmup',
        'Number of warmup epoch, slowly increasing KL weight to 1.', 400
  # ====== data ====== #
  ).add('-y-weight',
        'weight for supervised objective, only apply for semi-supervised system', 1
  ).add('--no-batchnorm', 'turn-off batch normalization', False
  ).add('--monitor', 'enable monitoring every epoch', False
  ).add('--no-analytic', 'turn off: analytic KL for variational inference', False
  ).add('--override', 'override previous experiments', False
  ).add('-nprocess', 'number or process to run multiple experiments at once', 1
  ).parse()
  args['batchnorm'] = not args['no_batchnorm']
  args['analytic'] = not args['no_analytic']
  n_process = int(args['nprocess'])
  del args['nprocess']
  assert n_process > 0
  # ====== post-processing ====== #
  from six import string_types
  preprocessed = {}
  for k, v in args.items():
    assert not isinstance(v, (tuple, list)), \
    "Invalid argument format, name='%s' value='%s'" % (k, str(v))
    if isinstance(v, string_types):
      v = v.split(',')
      if len(v) == 0:
        v = ''
      elif len(v) == 1:
        v = v[0]
      else:
        v = [int(i) if i.isdigit() else
            (float(i) if '.' != i and re.match('^[\+-]?\d*\.\d*$', i)
             else i)
            for i in v]
    preprocessed[k] = v
  return preprocessed, n_process

def train(model='vae', ds='pbmc8k_ly', dispersion='gene-cell',
          xnorm='log', ximpu=0.25, tnorm='raw',
          ynorm='prob',
          xclip=0, yclip=0,
          xdist='zinb', ydist='bernoulli', zdist='normal', cdist='binomial',
          lr=1e-4, epoch=128, batch=64, nsample=1, ps=0.8,
          hdim=128, zdim=32, nlayer=2,
          xdrop=0.3, edrop=0, ddrop=0, zdrop=0,
          batchnorm=True, analytic=True,
          kl_weight=1., warmup=400, y_weight=10.,
          monitor=False, override=False, **kwargs):
  import os
  os.environ['ODIN'] = 'gpu,float32,seed=5218'
  import shutil
  import pickle

  from odin.utils import (ctext, stdio)

  from sisua.data import (get_dataset, EXP_DIR)
  if model == 'scvi':
    from sisua.inference import InferenceSCVI as Inference
  elif model == 'dca':
    from sisua.inference import InferenceDCA as Inference
  else:
    from sisua.inference import Inference
  # ====== validate the arguments ====== #
  assert (0 < nlayer < 100 and
          0 < hdim < 1000 and
          0 < zdim < 1000)
  assert (0 <= ximpu < 1)
  assert (0 <= xdrop < 1 and
          0 <= edrop < 1 and
          0 <= zdrop < 1 and
          0 <= ddrop < 1)
  assert (xclip >= 0 and yclip >= 0)
  assert 0.0 < ps <= 1.0, \
      "`ps` value must be > 0 and <= 1, but given: %f" % float(ps)
  # ===========================================================================
  # Loading data
  # ===========================================================================
  DS_NAME = str(ds)
  (ds, gene_ds, prot_ds) = get_dataset(ds, override=False)
  # ====== data for training and testing ====== #
  X_train = gene_ds.get_data(data_type='train')
  y_train = prot_ds.get_data(data_type='train')
  X_test = gene_ds.get_data(data_type='test')
  y_test = prot_ds.get_data(data_type='test')
  # ===========================================================================
  # Training the network
  # ===========================================================================
  infer = Inference(
      gene_dim=X_train.shape[1],
      prot_dim=None if model in ('vae', 'scvi', 'dca')
      else y_train.shape[1],
      model=model, dispersion=dispersion,
      xnorm=xnorm, tnorm=tnorm, ynorm=ynorm,
      xclip=xclip, yclip=yclip,
      xdist=xdist, ydist=ydist, zdist=zdist,
      xdrop=xdrop, edrop=edrop, zdrop=zdrop, ddrop=ddrop,
      hdim=hdim, zdim=zdim, nlayer=nlayer,
      batchnorm=batchnorm, analytic=analytic,
      kl_weight=kl_weight, warmup=warmup, y_weight=y_weight,
      dataset=DS_NAME)
  infer.fit(X=X_train, y=y_train,
            supervised_percent=ps, validation_percent=0.1,
            corruption_rate=ximpu, corruption_dist=cdist,
            n_mcmc_samples=nsample,
            batch_size=batch, n_epoch=epoch, learning_rate=lr,
            monitoring=False, fig_dir=None,
            detail_logging=True)
  # ===========================================================================
  # Save the trained model
  # ===========================================================================
  BASE_DIR = os.path.join(EXP_DIR, ds.name)
  if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

  MODEL_DIR = os.path.join(BASE_DIR, infer.id)
  if bool(override) and os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  # ====== print some log ====== #
  stdio(os.path.join(MODEL_DIR, 'init.log'))
  print(ctext("Model ID:", 'cyan'), ctext(infer.id, 'lightcyan'))
  print(ctext("Base dir:", 'cyan'), ctext(BASE_DIR, 'lightcyan'))
  with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(infer, f)
  # ===========================================================================
  # Evaluation and make prediction
  # ===========================================================================
  print('\n\n')
  stdio(os.path.join(MODEL_DIR, 'score.log'))
  for name, x, y in zip(("Train", "Test"),
                        (X_train, X_test),
                        (y_train, y_test)):
    scores = infer.score(x, y, n_mcmc_samples=100)
    print("======== %s ========" % name)
    for k, v in scores.items():
      print("  ", '%-32s' % k.split(":")[0].split('/')[-1], v)
# ===========================================================================
# Calling the main
# ===========================================================================
def main():
  kw, n_process = get_arguments()
  from sisua import set_verbose
  set_verbose(True)
  multiple_args = [[(i, t) for t in j] for i, j in kw.items()
                   if isinstance(j, (tuple, list))]
  if len(multiple_args) > 0:
    from odin.utils import ctext
    from multiprocessing import Process

    multiple_args = list(product(*multiple_args))
    print("Start running %s experiments ..." %
      ctext(len(multiple_args), 'lightyellow'))

    runned_configs = {}

    processes = []
    for args in multiple_args:
      print("  \nConfigurations {%s}" %
        ', '.join(['%s:%s' % (i, ctext(j, 'cyan'))
                   for i, j in args]))
      # all new kwargs
      tmp = dict(kw)
      tmp.update(args)
      # special processing for unsupervised models
      if tmp['model'] in ('dca', 'scvi', 'vae'):
        tmp['ynorm'] = 'prob'
        tmp['yclip'] = 0
        tmp['ydist'] = 'bernoulli'
        tmp['ps'] = 0.8
        tmp['y_weight'] = 1
      # don't run duplicated experiments
      footprint = ','.join([str(i) + ':' + str(j)
                            for i, j in sorted(tmp.items())])
      if footprint in runned_configs:
        continue
      else:
        runned_configs[footprint] = 1
      # create new process for the training task
      processes.append(Process(target=train, kwargs=tmp))
      if len(processes) >= n_process:
        [p.start() for p in processes]
        [p.join() for p in processes]
        processes = []
    # don't forget to run the remains
    if len(processes) > 0:
      [p.start() for p in processes]
      [p.join() for p in processes]
  else:
    train(**kw)

if __name__ == '__main__':
  main()

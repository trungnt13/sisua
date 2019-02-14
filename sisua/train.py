from __future__ import print_function, division, absolute_import

def get_arguments():
  from odin.utils import ArgController
  args = ArgController(
  ).add('-model',
        'name for model, specified in models_.py',
        'vae'
  ).add('-ds',
        'name of the dataset: mnist, pbmc citeseq, pbmc 10xPP, cbmc, etc.',
        'facs_7'
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
  # ====== data ====== #
  ).add('--no-batchnorm', 'turn-off batch normalization', False
  ).add('--count-sum', 'introducing count-sum to the latent space', False
  ).add('--constraint', 'introducing count-sum to the reconstruction distribution', False
  ).add('--monitor', 'enable monitoring every epoch', False
  ).add('--no-iw', 'turn off: important weight sampling', False
  ).add('--no-analytic', 'turn off: analytic KL for variational inference', False
  ).add('--override', 'override previous experiments', False
  ).parse()
  args['batchnorm'] = not args['no_batchnorm']
  args['iw'] = not args['no_iw']
  args['analytic'] = not args['no_analytic']
  return args

def train(model='vae', ds='facs_7',
          xnorm='log', ximpu=0.1, tnorm='raw', ynorm='prob', yimpu=0,
          xclip=0, yclip=0,
          xdist='zinb', ydist='bernoulli', zdist='normal', cdist='uniform',
          lr=1e-4, epoch=128, batch=32, nsample=1, ps=0.8,
          hdim=128, zdim=32, nlayer=2,
          xdrop=0.3, edrop=0, ddrop=0, zdrop=0,
          batchnorm=True, count_sum=False, constraint=False, iw=True, analytic=True,
          monitor=False, override=False, **kwargs):
  import os
  os.environ['ODIN'] = 'gpu,float32,seed=5218'
  import shutil
  import pickle

  from odin.utils import (ctext, stdio)

  from sisua.data import (get_dataset, EXP_DIR)
  from sisua.inference import Inference
  # ====== validate the arguments ====== #
  assert (0 < nlayer < 100 and
          0 < hdim < 1000 and
          0 < zdim < 1000)
  assert (0 <= ximpu < 1 and
          0 <= yimpu < 1)
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
  X_train = gene_ds.get_data(data_type='train',
    dropout=ximpu, distribution=cdist)
  y_train = prot_ds.get_data(data_type='train',
    dropout=yimpu, distribution=cdist)
  X_test = gene_ds.get_data(data_type='test',
    dropout=ximpu, distribution=cdist)
  y_test = prot_ds.get_data(data_type='test',
    dropout=yimpu, distribution=cdist)
  # ===========================================================================
  # model identify
  # ===========================================================================
  # [0] model name
  # [1] feat name
  # [2] percentage of supervised data
  # [3] latent dimension
  # [4] use mouse genes or not
  CORRUPTION_DIR = os.path.join(EXP_DIR, cdist + '%.2d' % (ximpu * 100))
  if not os.path.exists(CORRUPTION_DIR):
    os.mkdir(CORRUPTION_DIR)

  MODEL_NAME = str(model).strip().lower()
  MODEL_ID = '_'.join([
      MODEL_NAME,
      'X%s%d%s' % (xnorm, int(max(xclip, 0)), xdist),
      'Y%s%d%s' % (ynorm, int(max(yclip, 0)), ydist),
      'T%s' % tnorm,
      'Z' + str(zdist),
      'spvs%.3d' % (ps * 100),
      'net%.2d%.3d%.3d' % (nlayer, hdim, zdim),
      'drop%.2d%.2d%.2d%.2d' % (xdrop * 100,
                                edrop * 100,
                                zdrop * 100,
                                ddrop * 100),
      'alytcT' if bool(analytic) else 'alytcF',
      'iwT' if bool(iw) else 'iwF',
      'bnormT' if bool(batchnorm) else 'bnormF',
      'cntsmT' if bool(count_sum) else 'cntsmF',
      'cstrnT' if bool(constraint) else 'cstrnF',
  ])
  BASE_DIR = os.path.join(CORRUPTION_DIR, ds.name)
  if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

  MODEL_DIR = os.path.join(BASE_DIR, MODEL_ID)
  if bool(override) and os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  # ====== print some log ====== #
  stdio(os.path.join(MODEL_DIR, 'init.log'))
  print(ctext("Model ID:", 'cyan'), ctext(MODEL_ID, 'lightred'))
  print(ctext("Base dir:", 'cyan'), ctext(BASE_DIR, 'lightred'))
  # ===========================================================================
  # Training the network
  # ===========================================================================
  stdio(os.path.join(MODEL_DIR, 'train.log'))
  infer = Inference(model=MODEL_NAME,
                    hdim=hdim, zdim=zdim, nlayer=nlayer,
                    xnorm=xnorm, tnorm=tnorm, ynorm=ynorm,
                    xclip=xclip, yclip=yclip,
                    xdist=xdist, ydist=ydist, zdist=zdist,
                    xdrop=xdrop, edrop=edrop, zdrop=zdrop, ddrop=ddrop,
                    batchnorm=batchnorm, count_sum=count_sum,
                    analytic=analytic, iw=iw,
                    cellsize_normalize_factor=gene_ds.cell_median,
                    # store some extra arguments
                    ximpu=ximpu, yimpu=yimpu, cdist=cdist, dataset=DS_NAME)
  infer.fit(X=X_train, y=y_train,
            supervised_percent=ps, validation_percent=0.1,
            n_mcmc_samples=nsample,
            batch_size=batch, n_epoch=epoch, learning_rate=lr,
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
    scores = infer.score(x, y, n_mcmc_samples=100)
    print("======== %s ========" % name)
    for k, v in scores.items():
      print("  ", '%-32s' % k.split(":")[0].split('/')[-1], v)
# ===========================================================================
# Calling the main
# ===========================================================================
if __name__ == '__main__':
  kw = get_arguments()
  from sisua import set_verbose
  set_verbose(True)
  train(**kw)

# This file will re-procedure running configurations of our experiments
from __future__ import print_function, division, absolute_import
import os
# set the experiment path to different folder so it does not
# override your experiments
os.environ['SISUA_EXP'] = '/tmp/bio_log'

from sisua import set_verbose
from sisua.train import train
from sisua.data import get_dataset

epoch_database = dict(
    cortex=250,
    pbmc=200,
    facs_5=400,
    facs_7=400,
    pbmc_citeseq=128,
    cbmc_citeseq=128,
)

def run(cdist):
  cdist = str(cdist).lower()
  assert cdist in ('uniform', 'binomial')

  for dataset_name in ['cortex', 'pbmc',
                       'facs_5', 'facs_7',
                       'pbmc_citeseq', 'cbmc_citeseq']:
    print(dataset_name, cdist)
    ds, gene, prot = get_dataset(dataset_name=dataset_name, override=False)
    for model_name in ['unsupervised', 'multitask', 'dualtask',
                       'vae',
                       'mlvae', 'movae',
                       'dlvae', 'dovae']:
      for xnorm in ['log', 'raw']:
        for ximpu in [0.1, 0.2, 0.3, 0.5, 0.8]:
          for ps in [0.1, 0.4, 0.8, 1.0]:
            for ynorm, ydist in [('prob', 'bernoulli'),
                                 ('raw', 'nb')]:
              # skip of labeled data is just binary
              if prot.is_binary and ydist == 'nb':
                continue
              # skip for unsupervised models
              if ynorm == 'raw' and \
              model_name in ('unsupervised', 'vae', 'multitask', 'dualtask'):
                continue
              # print some log
              print("  ", model_name,
                    "xnorm:%s" % xnorm, ximpu,
                    "ynorm:%s" % ynorm, ydist, ps)
              train(model=model_name, ds=dataset_name,
                    xclip=0, xnorm=xnorm, ximpu=ximpu, xdist='zinb',
                    yclip=0, ynorm=ynorm, yimpu=0, ydist=ydist,
                    tnorm='raw',
                    zdist='normal', cdist=cdist,
                    lr=1e-4, epoch=epoch_database.get(dataset_name, 128),
                    batch=32, nsample=1, ps=ps,
                    hdim=128, zdim=32, nlayer=2,
                    xdrop=0.3, edrop=0, ddrop=0, zdrop=0,
                    batchnorm=True, count_sum=False, constraint=False,
                    iw=True, analytic=True,
                    monitor=False, override=True)

if __name__ == '__main__':
  from odin.utils import ArgController
  args = ArgController(
  ).add('cdist', 'corruption distribution to run'
  ).parse()
  set_verbose(False)
  run(cdist=args.cdist)

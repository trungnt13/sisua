from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import tensorflow as tf

from odin.bay.vi import Criticizer
from odin.bay.vi.downstream_metrics import predictive_strength
from odin.stats import sparsity_percentage
from odin.utils import ArgController
from sisua import (MARKER_ADTS, MARKER_GENES, SISUA, Posterior, RandomVariable,
                   SingleCellModel, SingleCellOMIC, SisuaExperimenter,
                   get_dataset)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# Arguments
# ===========================================================================
args = ArgController(\
).add("-m", "The model alias", "sisua" \
).add("-ds1", "The first dataset", "8kx" \
).add("-ds2", "The second dataset, if empty, only analyze the first one", "" \
).add("-bs", "batch size", 8 \
).add("--override", "remove exists folders", False \
).parse()

MODEL = str(args.m)
DS1 = str(args.ds1)
DS2 = str(args.ds2)
BATCH_SIZE = int(args.bs)
assert len(DS1) > 0, "-ds1 option for the first dataset must be provided."

######## paths
se = SisuaExperimenter().eval_mode()
result_dir = se.get_result_dir()

path1 = os.path.join(result_dir, f"{MODEL}{DS1}")
# train on ds1, test on ds2
path12 = os.path.join(result_dir, f"{MODEL}{DS1}_{DS2}")
# train on ds2, test on ds1
path21 = os.path.join(result_dir, f"{MODEL}{DS2}_{DS1}")

if args.override:
  for p in [path1, path12, path21]:
    if os.path.exists(p):
      print(f"Override path '{p}'")
      shutil.rmtree(p)

if not os.path.exists(path1):
  os.makedirs(path1)
if len(DS2) > 0:
  if not os.path.exists(path12):
    os.makedirs(path12)
  if not os.path.exists(path21):
    os.makedirs(path21)

# ===========================================================================
# Load the model and dataset
# ===========================================================================
hash1, cfg1, m1 = se.get_models(f"dataset.name={DS1} model.name={MODEL}",
                                load_models=True,
                                return_hash=True)
test1: SingleCellOMIC = m1.test
vae1: SingleCellModel = m1.model
is_semi = vae1.is_semi_supervised

if len(DS2) > 0:
  hash2, cfg2, m2 = se.get_models(f"dataset.name={DS2} model.name={MODEL}",
                                  load_models=True,
                                  return_hash=True)
  test2: SingleCellOMIC = m2.test
  vae2: SingleCellModel = m2.model
else:
  test2 = None
  vae2 = None
  cfg2 = None
  hash2 = None

# ===========================================================================
# Create the posterior
# ===========================================================================
kw = dict(batch_size=BATCH_SIZE, verbose=True)

# mapping from:
# tuple (save_path, train_dsname, test_dsname) -> SingleCellModel
all_posteriors = {}
if vae2 is None:
  post = Posterior(vae1, test1, name=f"{MODEL}{DS1}", **kw)
  all_posteriors[(path1, test1.name, test1.name)] = post
else:
  all_posteriors[(path12, test1.name,
                  test2.name)] = Posterior(vae1,
                                           test2,
                                           name=f"{MODEL}{DS1}_{DS2}",
                                           **kw)
  all_posteriors[(path21, test2.name,
                  test1.name)] = Posterior(vae2,
                                           test1,
                                           name=f"{MODEL}{DS2}_{DS1}",
                                           **kw)
# ===========================================================================
# Evaluating
# ===========================================================================
for (path, train_ds, test_ds), post in all_posteriors.items():
  print(f"Evaluate model:'{MODEL}' train:'{train_ds}' test:'{test_ds}', "
        f"save results at: '{path}'")
  post: Posterior
  name = os.path.basename(path)
  ### calculateing the scores
  # for om in post.output_omics:
  #   llk = post.cal_llk(omic=om)
  # print(post.cal_dci())
  # print(post.cal_mig())
  # print(post.cal_factorvae())
  # print(post.cal_betavae())
  # print(post.cal_imputation_scores())
  # print(post.cal_mutual_information())
  # print(post.cal_pearson())
  # print(post.cal_spearman())
  ### plotting the figures
  post.plot_disentanglement_scatter('proteomic')
  post.plot_disentanglement('proteomic')
  if is_semi:
    post.plot_disentanglement_scatter('iproteomic')
    post.plot_disentanglement('iproteomic')
    post.plot_series('proteomic', 'iproteomic', MARKER_ADTS, MARKER_ADTS, False,
                     False)
  post.save_figures(path, separate_files=True, clear_figures=True, verbose=True)
  #
  post.plot_scatter('latent', 'proteomic')
  post.plot_violins('transcriptomic', 'proteomic')
  post.plot_violins('itranscriptomic', 'proteomic')
  post.plot_heatmap('transcriptomic', 'proteomic')
  post.plot_heatmap('itranscriptomic', 'proteomic')
  post.plot_distance_heatmap('transcriptomic', 'proteomic')
  post.plot_distance_heatmap('itranscriptomic', 'proteomic')
  if is_semi:
    post.plot_scatter('latent', 'iproteomic')
    post.plot_violins('transcriptomic', 'iproteomic')
    post.plot_violins('itranscriptomic', 'iproteomic')
    post.plot_heatmap('transcriptomic', 'iproteomic')
    post.plot_heatmap('itranscriptomic', 'iproteomic')
    post.plot_distance_heatmap('transcriptomic', 'iproteomic')
    post.plot_distance_heatmap('itranscriptomic', 'iproteomic')
  post.save_figures(path, separate_files=True, clear_figures=True, verbose=True)
  #
  post.plot_correlation_matrix('transcriptomic', 'proteomic')
  post.plot_correlation_matrix('itranscriptomic', 'proteomic')
  post.plot_correlation_scatter('transcriptomic', 'proteomic')
  post.plot_correlation_scatter('itranscriptomic', 'proteomic')
  if is_semi:
    post.plot_correlation_matrix('transcriptomic', 'iproteomic')
    post.plot_correlation_matrix('itranscriptomic', 'iproteomic')
    post.plot_correlation_scatter('transcriptomic', 'iproteomic')
    post.plot_correlation_scatter('itranscriptomic', 'iproteomic')
  post.save_figures(path, separate_files=True, clear_figures=True, verbose=True)

  # post.plot_confusion_matrix('progenitor', 'iprogenitor')

### Examples:
# python evaluate.py -m sisua -ds1 8kx -ds2 eccx
# python evaluate.py -m vae -ds1 8kx -ds2 eccx
# python evaluate.py -m dca -ds1 8kx -ds2 eccx
# python evaluate.py -m scale -ds1 8kx -ds2 eccx
# python evaluate.py -m sisua -ds1 8kx -ds2 vdj4x
# python evaluate.py -m vae -ds1 8kx -ds2 vdj4x
# python evaluate.py -m dca -ds1 8kx -ds2 vdj4x
# python evaluate.py -m scale -ds1 8kx -ds2 vdj4x
# python evaluate.py -m sisua -ds1 eccx -ds2 vdj4x
# python evaluate.py -m vae -ds1 eccx -ds2 vdj4x
# python evaluate.py -m dca -ds1 eccx -ds2 vdj4x
# python evaluate.py -m scale -ds1 eccx -ds2 vdj4x

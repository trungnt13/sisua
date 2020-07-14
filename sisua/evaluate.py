from __future__ import absolute_import, division, print_function

import os
import shutil
import traceback

import numpy as np
import tensorflow as tf

from odin.bay.vi import Criticizer
from odin.bay.vi.downstream_metrics import predictive_strength
from odin.stats import sparsity_percentage
from odin.utils import ArgController, catch_warnings_ignore
from sisua import (MARKER_ADTS, MARKER_GENES, OMIC, PROTEIN_PAIR_NEGATIVE,
                   SISUA, Posterior, RandomVariable, SingleCellModel,
                   SingleCellOMIC, SisuaExperimenter, get_dataset)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# Arguments
# ===========================================================================
args = ArgController(\
).add("model", "The model alias" \
).add("ds1", "The first dataset", "eccx" \
).add("ds2", "The second dataset, if empty, only analyze the first one", "" \
).add("-bs", "batch size", 4 \
).add("--override", "remove exists folders", False \
).add("--score", "re-calculating the model scores", False \
).add("--plot", "plotting the analysis", False \
).parse()

MODEL = str(args.model)
DS1 = str(args.ds1)
DS2 = str(args.ds2)
BATCH_SIZE = int(args.bs)
assert len(DS1) > 0, "-ds1 option for the first dataset must be provided."

SCORE = args.score
PLOT = args.plot

if not any([SCORE, PLOT]):
  SCORE = True
  PLOT = True
print("Scoring :", SCORE)
print("Plotting:", PLOT)
######## paths
se = SisuaExperimenter().eval_mode()
result_dir = se.get_result_dir()

path1 = os.path.join(result_dir, f"{MODEL}_{DS1}")
# train on ds1, test on ds2
path12 = os.path.join(result_dir, f"{MODEL}_{DS1}_{DS2}")
# train on ds2, test on ds1
path21 = os.path.join(result_dir, f"{MODEL}_{DS2}_{DS1}")

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
  post = Posterior(vae1, test1, name=f"{MODEL}_{DS1}", **kw)
  all_posteriors[(path1, DS1, DS1)] = post
else:
  all_posteriors[(path12, DS1, DS2)] = Posterior(vae1,
                                                 test2,
                                                 name=f"{MODEL}_{DS1}_{DS2}",
                                                 **kw)
  all_posteriors[(path21, DS2, DS1)] = Posterior(vae2,
                                                 test1,
                                                 name=f"{MODEL}_{DS2}_{DS1}",
                                                 **kw)


# ===========================================================================
# Helper
# ===========================================================================
def scoring(path: str, train_ds: str, test_ds: str, post: Posterior):
  name = os.path.basename(path)
  scores = dict(name=name)
  # incompatible shape
  for om in post.output_omics:
    llk = post.cal_llk(omic=om)
    scores.update(llk)
  scores.update(post.cal_dci())
  scores.update(post.cal_mig())
  scores.update(post.cal_factorvae())
  scores.update(post.cal_imputation_scores())
  scores.update(post.cal_mutual_information())
  scores.update(post.cal_pearson())
  scores.update(post.cal_spearman())
  se.write(table=f"scores_{test_ds}", unique='name', replace=True, **scores)


def plotting(path: str, train_ds: str, test_ds: str, post: Posterior):
  input_omics = [OMIC.parse(i) for i in post.input_omics]
  output_omics = [OMIC.parse(i) for i in post.output_omics]
  _save = lambda: post.save_figures(
      path, separate_files=True, clear_figures=True, verbose=True)
  post.plot_learning_curves()
  _save()
  for is_binary, factor_omic in [
      (False, OMIC.proteomic),
      (True, OMIC.disease),
      (True, OMIC.progenitor),
      (True, OMIC.celltype),
  ]:
    print(" - plotting factor:", factor_omic)
    if factor_omic not in input_omics:
      continue
    imputed_omic = OMIC.parse(f'i{factor_omic.name}')
    # compare original vs prediction
    if imputed_omic in output_omics:
      post.plot_series(factor_omic,
                       imputed_omic,
                       factor_omic.markers,
                       factor_omic.markers,
                       log1=False,
                       log2=False)
    # proteomic disentanglement
    if not is_binary:
      post.plot_disentanglement(factor_omic)
      post.plot_disentanglement_scatter(factor_omic,
                                        pairs=PROTEIN_PAIR_NEGATIVE)
      if factor_omic in output_omics:
        post.plot_disentanglement(imputed_omic)
        post.plot_disentanglement_scatter(imputed_omic,
                                          pairs=PROTEIN_PAIR_NEGATIVE)
      _save()
    # scatter plot
    post.plot_scatter('latent',
                      color_by=factor_omic,
                      dimension_reduction='tsne')
    post.plot_scatter('latent',
                      color_by=factor_omic,
                      dimension_reduction='umap')
    if factor_omic in output_omics:
      post.plot_scatter('latent',
                        color_by=imputed_omic,
                        dimension_reduction='tsne')
      post.plot_scatter('latent',
                        color_by=imputed_omic,
                        dimension_reduction='umap')
    _save()
    # heatmap and violin
    post.plot_violins('transcriptomic', factor_omic)
    post.plot_violins('itranscriptomic', factor_omic)
    post.plot_heatmap('transcriptomic', factor_omic)
    post.plot_heatmap('itranscriptomic', factor_omic)
    if factor_omic in output_omics:
      post.plot_violins('transcriptomic', imputed_omic)
      post.plot_violins('itranscriptomic', imputed_omic)
      post.plot_heatmap('transcriptomic', imputed_omic)
      post.plot_heatmap('itranscriptomic', imputed_omic)
    _save()
    # for binary factor
    if is_binary:
      post.plot_distance_heatmap('transcriptomic', factor_omic)
      post.plot_distance_heatmap('itranscriptomic', factor_omic)
      if factor_omic in output_omics:
        post.plot_distance_heatmap('transcriptomic', imputed_omic)
        post.plot_distance_heatmap('itranscriptomic', imputed_omic)
    _save()
    # correlation matrix
    if not is_binary:
      post.plot_correlation_matrix('transcriptomic', factor_omic)
      post.plot_correlation_matrix('itranscriptomic', factor_omic)
      post.plot_correlation_scatter('transcriptomic', factor_omic)
      post.plot_correlation_scatter('itranscriptomic', factor_omic)
      if factor_omic in output_omics:
        post.plot_correlation_matrix('transcriptomic', imputed_omic)
        post.plot_correlation_matrix('itranscriptomic', imputed_omic)
        post.plot_correlation_scatter('transcriptomic', imputed_omic)
        post.plot_correlation_scatter('itranscriptomic', imputed_omic)
      _save()
  # post.plot_confusion_matrix('progenitor', 'iprogenitor')


# ===========================================================================
# Evaluating
# ===========================================================================
for (path, train_ds, test_ds), post in all_posteriors.items():
  print(f"Evaluate model:'{MODEL}' train:'{train_ds}' test:'{test_ds}', "
        f"save results at: '{path}'")
  post: Posterior
  name = os.path.basename(path)
  with catch_warnings_ignore(UserWarning):
    ### calculateing the scores
    if SCORE:
      try:
        scoring(path, train_ds, test_ds, post)
      except Exception as e:
        traceback.print_exc()
        print(e)
    ### plotting the figures
    if PLOT:
      try:
        plotting(path, train_ds, test_ds, post)
      except Exception as e:
        traceback.print_exc()
        print(e)

### Examples:
# python evaluate.py sisua -ds1 8kx
# python evaluate.py sisua -ds1 eccx
# python evaluate.py sisua -ds1 8kly
# python evaluate.py sisua -ds1 8klyall
# python evaluate.py sisua -ds1 mpalx
# python evaluate.py sisua -ds1 callx
# python evaluate.py sisua -ds1 vdj4x

# python evaluate.py sisua -ds1 8kx -ds2 eccx
# python evaluate.py vae -ds1 8kx -ds2 eccx
# python evaluate.py dca -ds1 8kx -ds2 eccx
# python evaluate.py sisua -ds1 8kx -ds2 mpalx
# python evaluate.py vae -ds1 8kx -ds2 mpalx
# python evaluate.py dca -ds1 8kx -ds2 mpalx
# python evaluate.py sisua -ds1 8kx -ds2 vdj4x
# python evaluate.py vae -ds1 8kx -ds2 vdj4x
# python evaluate.py dca -ds1 8kx -ds2 vdj4x

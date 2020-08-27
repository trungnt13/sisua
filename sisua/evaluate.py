from __future__ import absolute_import, division, print_function

import os
import shutil
import sys
import traceback
from io import StringIO
from itertools import product
from multiprocessing import Process

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

SE = SisuaExperimenter().eval_mode()


def robust_run(method_name, log_text, fn, *args, **kwargs):
  r""" Run an evaluation function and catch exception without interupting the
  execution """
  assert callable(fn)
  with catch_warnings_ignore(UserWarning):
    try:
      fn(*args, **kwargs)
    except Exception as e:
      text = StringIO()
      traceback.print_exception(*sys.exc_info(),
                                limit=None,
                                file=text,
                                chain=True)
      text.seek(0)
      text = text.read().strip()
      text += f"\n{e}"
      SE.write_error(traceback=text, method_name=method_name, config=log_text)


# ===========================================================================
# Helper
# ===========================================================================
def scoring(post: Posterior, path: str, train_ds: str, test_ds: str):
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
  SE.write(table=f"scores_{test_ds}", unique='name', replace=True, **scores)


def plotting(post: Posterior, path: str, train_ds: str, test_ds: str):
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
def main(model,
         ds1,
         ds2,
         batch_size,
         score_enable,
         plot_enable,
         override=False):
  print("Start evaluation:")
  print(f" - model     : {model}")
  print(f" - dataset1  : {ds1}")
  print(f" - dataset2  : {ds2}")
  print(f" - batch_size: {batch_size}")
  print(f" - override  : {override}")
  print(f" - plot:{plot_enable} score:{score_enable}")
  result_dir = SE.get_result_dir()
  if len(ds2) == 0:
    outpath = os.path.join(result_dir, f"{model}_{ds1}")
  else:
    outpath = os.path.join(result_dir, f"{model}_{ds1}_{ds2}")
  # overriding exist paths
  if override and os.path.exists(outpath):
    print(f"Override path '{outpath}'")
    shutil.rmtree(outpath)
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  ### Load the model and dataset
  hash1, cfg1, m1 = SE.get_models(f"dataset.name={ds1} model.name={model}",
                                  load_models=True,
                                  return_hash=True)[0]
  test1: SingleCellOMIC = m1.test
  vae1: SingleCellModel = m1.model
  is_semi = vae1.is_semi_supervised
  if len(ds2) > 0:
    hash2, cfg2, m2 = SE.get_models(f"dataset.name={ds2} model.name={model}",
                                    load_models=True,
                                    return_hash=True)[0]
    test2: SingleCellOMIC = m2.test
    vae2: SingleCellModel = m2.model
  else:
    test2 = None
    vae2 = None
    cfg2 = None
    hash2 = None
  # Create the posterior
  kw = dict(batch_size=batch_size, verbose=True)
  # mapping from:
  if vae2 is None:
    posterior = Posterior(vae1, test1, name=f"{model}_{ds1}", **kw)
  else:
    posterior = Posterior(vae1, test2, name=f"{model}_{ds1}_{ds2}", **kw)
  ### running the evaluation
  train_ds = ds1
  test_ds = ds2
  with catch_warnings_ignore(UserWarning):
    # calculateing the scores
    if score_enable:
      robust_run("evaluate_scoring",
                 f"model:{model} train:{train_ds} test:{test_ds}", scoring,
                 posterior, outpath, train_ds, test_ds)
    # plotting the figures
    if plot_enable:
      robust_run("evaluate_plotting",
                 f"model:{model} train:{train_ds} test:{test_ds}", plotting,
                 posterior, outpath, train_ds, test_ds)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  args = ArgController(\
  ).add("model", "The model alias, multiple model separated by comma" \
  ).add("ds1", "The first dataset", "eccx" \
  ).add("-ds2", "The second dataset, if empty, only analyze the first one", "" \
  ).add("-bs", "batch size", 4 \
  ).add("--score", "re-calculating the model scores", False \
  ).add("--plot", "plotting the analysis", False \
  ).add("--override", "remove exists folders", False \
  ).add("--only-cross", "Only do cross dataset experiments", False \
  ).parse()
  # preprocess the arguments
  models = args.model.split(",")
  ds1s = args.ds1.split(",")
  ds2s = args.ds2.split(",")
  assert len(ds1s) > 0 and len(ds1s[0])> 0,\
    "-ds1 option for the first dataset must be provided."
  # evaluation modes
  plot = bool(args.plot)
  score = bool(args.score)
  if not (plot or score):
    plot = True
    score = True

  def _eval_fn(model_name, ds1_name, ds2_name):
    main(model=model_name,
         ds1=ds1_name,
         ds2=ds2_name,
         batch_size=int(args.bs),
         score_enable=score,
         plot_enable=plot,
         override=bool(args.override))

  # iterate over all possibility
  configs = list(
      set([(m, d1, "" if d1 == d2 else d2)
           for m, d1, d2 in product(models, ds1s, ds2s)]))
  if args.only_cross:
    configs = [(m, d1, d2) for m, d1, d2 in configs if len(d2) > 0]
  np.random.shuffle(configs)
  print(f"Running {len(configs)} configurations:")
  for cfg in configs:
    print(f" - {', '.join(cfg)}")
  # only run if there is config
  if len(configs) > 0:
    proc = [Process(target=_eval_fn, args=(m, d1, d2)) for m, d1, d2 in configs]
    for p in proc:
      try:
        p.start()
        p.join()
        if p.is_alive():
          p.terminate()
      except Exception as e:
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

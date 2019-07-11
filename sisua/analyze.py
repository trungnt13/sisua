from __future__ import print_function, division, absolute_import
import os
import shutil

import pandas as pd

from odin.utils import UnitTimer, ctext, ArgController

from sisua.analysis import Posterior, ResultsSheet, get_all_posteriors
from sisua.data.path import EXP_DIR

# ===========================================================================
# Helper
# ===========================================================================
def get_arguments():
  args = ArgController(
  ).add('ds', 'dataset for evaluation'
  ).add('-model', 'name of all model for evaluation, e.g. vae,movae,dca,scvi', 'vae'
  ).add('-ynorm', 'specific normalization mode for protein', ''
  ).add('-yweight', 'specific weight for protein, e.g. 1,10,100', ''
  ).add('-incl', 'list of all keywords for inclusion', ''
  ).add('-excl', 'list of all keywords for exclusion', ''
  ).add('-outpath', 'Output folder to save all the results', ''
  ).parse()

  # ====== checking dataset ====== #
  dataset_name = args.ds

  # ====== checking models ====== #
  models = [i for i in args.models.split(',') if len(i) > 0]
  assert len(models) > 0

  # ====== checking keywords ====== #
  incl = [i for i in args.incl.split(',') if len(i) > 0]
  excl = [i for i in args.incl.split(',') if len(i) > 0]

  # ====== checking protein normalization ====== #
  ynorm = ''
  if args.ynorm == 'prob':
    ynorm = 'Yprob0bernoulli'
  elif args.ynorm == 'raw':
    ynorm = 'Yraw0nb'

  yweight = str(args.yweight)
  yweight = ['%.2g' % float(i)
             for i in yweight.split(',')
             if len(i) > 0]

  # ====== checking output path ====== #
  if args.outpath == '':
    if not os.path.exists('/tmp/results'):
      os.mkdir('/tmp/results')
    outpath = os.path.join('/tmp/results', dataset_name.replace('_', ''))
  else:
    outpath = args.outpath
  outpath = outpath + '_' + '_'.join(models)
  if len(ynorm) > 0:
    outpath += '_' + ynorm
  # incl = ['080spvs1e+02']

  return dataset_name, models, incl, excl, ynorm, yweight, outpath

# ===========================================================================
# Main analyzer
# ===========================================================================
def analyze(dataset_name, models, incl, excl,
            ynorm, yweight, outpath):
  dataset_path = os.path.join(EXP_DIR, dataset_name)
  assert os.path.exists(dataset_path), dataset_path

  if os.path.exists(outpath):
    shutil.rmtree(outpath)
  os.mkdir(outpath)
  score_path = os.path.join(outpath, 'scores.html')

  print("Dataset:", ctext(dataset_name, 'lightyellow'))
  print("Testing models:", ctext(models, 'lightyellow'))
  print("Protein:")
  print(" Yweight:", ctext(yweight, 'yellow'))
  print(" Ynorm:", ctext(ynorm, 'yellow'))
  print("Keywords:")
  print(" Include:", ctext(incl, 'yellow'))
  print(" Exclude:", ctext(excl, 'yellow'))
  print("Saving output at %s" % ctext(outpath, 'lightcyan'))
  print("Saving scores at %s" % ctext(score_path, 'lightcyan'))

  # ====== create the filtering functions ====== #
  def fn_filter(keywords):
    if not any(i in keywords for i in models):
      return False

    if not all(i in keywords for i in incl):
      return False

    if any(i in keywords for i in excl):
      return False

    # only continue with semi-supervised model
    if not ('scvi' in keywords or 'dca' in keywords or 'vae' in keywords):
      if len(ynorm) > 0:
        if keywords[3] != ynorm:
          return False

      if len(yweight) > 0:
        if not any(i == keywords[6].split('spvs')[-1]
                   for i in yweight):
          return False

    return True

  # ====== loading the Posteriors ====== #
  all_posteriors = get_all_posteriors(dataset_path,
      fn_filter=fn_filter,
      show_progress=True
  )

  # Extracting the report
  res = ResultsSheet(all_posteriors, verbose=True)
  print(res)

  # ====== scatter and series ====== #
  res.plot_learning_curves()

  res.plot_imputation_scatter(test=False, pca=True, color_by_library=True)
  res.plot_imputation_scatter(test=True, pca=True, color_by_library=True)

  res.plot_imputation_scatter(test=False, pca=True, color_by_library=False)
  res.plot_imputation_scatter(test=True, pca=True, color_by_library=False)

  res.plot_correlation_marker_pairs(test=False)
  res.plot_correlation_marker_pairs(test=True)

  res.plot_latents_binary_scatter(test=False)
  res.plot_latents_binary_scatter(test=True)

  res.save_plots(outpath, separate_files=True)

  # ====== save scores ====== #
  res.save_scores(score_path)

  # ====== plot scores ====== #
  res.plot_scores('cluster'
  ).plot_scores('pearson'
  ).plot_scores('spearman'
  ).plot_scores('imputation'
  ).plot_scores('classifier'
  )
  res.save_plots(outpath, separate_files=True)

# ===========================================================================
# Main
# ===========================================================================if __name__ == '__main__':
def main():
  args = get_arguments()
  analyze(*args)

if __name__ == '__main__':
  main()

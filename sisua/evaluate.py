from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from odin import search
from odin import visual as vs
from odin.bay import vi
from odin.exp import Experimenter
from odin.utils import get_formatted_datetime
from sisua.analysis import Posterior, ResultsSheet
from sisua.data import OMIC, get_dataset
from sisua.train import SisuaExperimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
# model, cfg = exp.search(dict(model="sisua"))[0]
# [f() for f in [lambda: i for i in range(10)]]

exp = SisuaExperimenter()
model, cfg = exp.sample_model({"model": "sisua", "dataset.name": "8kly"})
sco = get_dataset(model.fitted_dataset)
split = float(cfg.dataset.get('split', 0.8))
train, test = sco.split(0.98)
x = test.dimension_reduce(algo='tsne')
# test.plot_correlation(log=False, top=2, bottom=2, marker=True)
# test.plot_correlation_matrix(n_genes=None)
om1, om2 = sco.omics
varnames1 = sco.omic_varnames(om1)
varnames2 = sco.omic_varnames(om2)
# ====== test ====== #
pos = Posterior(scm=model, sco=test, batch_size=16, n_mcmc=10, verbose=True)
# pos.plot_cellsize_series(log=False)
crt = pos.create_criticizer(predicted=False, n_samples=100)
sco = pos.create_sco(keep_omic=OMIC.proteomic)
# crt.plot_uncertainty_scatter()
print(crt.cal_density_matrix())
exit()
crt.plot_histogram_heatmap(factors=['CD4', 'CD8'],
                           factor_bins=20,
                           n_codes_per_factor=None)
crt.save_figures()
exit()
# crt.cal_factorvae_score()
# crt.cal_betavae_score()
# crt.plot_histogram()
# crt.plot_uncertainty_statistics()
# crt.plot_code_factor_matrix()
# crt.plot_uncertainty_scatter()
# crt.cal_mutual_info_est()
# crt.cal_mutual_info_gap()
# crt.cal_separated_attr_predictability()
# crt.cal_dci_scores()
# crt.cal_total_correlation()
# crt.cal_importance_matrix()
# cor1 = crt.cal_correlation_matrix(method='pearson')
# cor2 = crt.cal_correlation_matrix(method='spearman')
# cor3 = crt.cal_correlation_matrix(method='lasso')
# train, test, ids = crt.cal_correlation_matrix(method='avg', decode=True)
# crt.cal_relative_disentanglement_strength()
#
sco.plot_divergence(X=OMIC.transcriptomic)
sco.plot_divergence(X=OMIC.latent, algo='tsne')
# sco.plot_scatter(X=OMIC.transcriptomic)
# sco.plot_scatter(X=OMIC.latent)
# sco.plot_correlation()
# sco.plot_correlation_matrix()
sco.save_figures('/tmp/tmp.pdf')
exit()
test.plot_heatmap(X=om1, groupby=om2, clustering='kmeans')
sco.plot_heatmap(X=om1, groupby=om2, clustering='kmeans')
#
sco = pos.create_sco(keep_omic=OMIC.transcriptomic)
test.plot_heatmap(X=om2, groupby=om1, clustering='kmeans')
sco.plot_heatmap(X=om2, groupby=om1, clustering='kmeans')
#
vs.plot_save()

# pos.plot_latents_risk(seed=1)
# pos.save_figures()
exit()

c = vi.Criticizer(model)
c.sample_batch(
    test.numpy(om1),
    test.numpy(om2),
    factor_name=varnames2,
    n_samples=(500, 200),  # train, test
    discretizing=False)
train, test = c.correlation_matrix()
ids1 = search.diagonal_bruteforce_search(train.T)
ids2 = search.diagonal_beam_search(train.T)
vs.plot_figure()
vs.plot_heatmap(train[ids1],
                xticklabels=c.factor_name,
                yticklabels=c.code_name[ids1])
vs.plot_figure()
vs.plot_heatmap(train[ids2],
                xticklabels=c.factor_name,
                yticklabels=c.code_name[ids2])
vs.plot_save()
exit()
c.plot_uncertainty_scatter(['CD4', 'CD8'], algo='tsne')
c.plot_uncertainty_statistics(['CD4', 'CD8'])
c.plot_code_factor_matrix(['CD4', 'CD8'])
c.save_figures('/tmp/figures/%s' % model.id)
exit()
# print(c.correlation_matrix())
# print(c.correlation_matrix(corr_type='pearson'))
# print(c.correlation_matrix(corr_type='lasso'))
c1 = c.conditioning(dict(CD8=lambda x: x > 6))
c2 = c.conditioning(dict(CD4=lambda x: x > 6))
vs.plot_figure(nrow=6, ncol=25)
nrow = 2
ncol = 1 + int(np.ceil(c.n_representations / 2))
bins = 40
plot = 1
code = 0
for row, name in enumerate(("CD8", "CD4")):
  index = c.index(name)
  for col in range(ncol):
    ax = plt.subplot(nrow, ncol, plot)
    if col == 0:
      x1 = c1.original_factors[0][:, index]
      x2 = c2.original_factors[0][:, index]
      title = name
    else:
      x1 = c1.representations_mean[0][:, code]
      x2 = c2.representations_mean[0][:, code]
      title = "Latent#%d" % code
      code += 1
    vs.plot_histogram(x1, bins=bins, ax=ax, color='red')
    vs.plot_histogram(x2, bins=bins, ax=ax, color='blue')
    ax.set_title(title)
    plt.tight_layout()
    plot += 1
vs.plot_save()
ztrain, ztest = c1.traversing(0)
# print(ztrain)
# c1.plot_histogram()
# c1.save_figures("/tmp/tmp1.pdf")
# c2.plot_histogram()
# c2.save_figures("/tmp/tmp2.pdf")
# print(c.total_correlation())
# print(c.mutual_info_gap())
# print(c.mutual_info_estimate())
# print(c.separated_attr_predictability())
# print(c.dci_scores())
exit()
# pos.plot_latents_uncertainty_scatter()

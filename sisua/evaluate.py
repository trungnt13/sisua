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
from sisua.data import get_dataset
from sisua.train import SisuaExperimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
# model, cfg = exp.search(dict(model="sisua"))[0]
# [f() for f in [lambda: i for i in range(10)]]

exp = SisuaExperimenter()
model, cfg = exp.sample_model({"model": "sisua", "dataset.name": "8kly"})
gene, prot = get_dataset(cfg.dataset.name)
split = float(cfg.dataset.get('split', 0.8))
x_train, x_test = gene.split(0.95)
if prot is not None:
  y_train, y_test = prot.split(0.95)

c = vi.Criticizer(model)
c.sample_batch(x_test.numpy(),
               y_test.numpy(),
               factor_name=prot.variable_name,
               n_samples=(500, 200),
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

pos = Posterior(scm=model,
                gene=x_test,
                protein=y_test,
                batch_size=16,
                n_mcmc=10)
pos.plot_latents_risk(seed=1)
pos.save_figures()
# pos.plot_latents_uncertainty_scatter()

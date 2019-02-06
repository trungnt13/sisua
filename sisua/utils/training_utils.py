from __future__ import print_function, division, absolute_import
import os
import colorsys
from numbers import Number
from collections import defaultdict

import numpy as np

from odin import visual as V
from odin.training import Callback

# ===========================================================================
# Helpers
# ===========================================================================
linestyles = ['-', '--', '-.', ':']

def _normalize_name(name):
  name = name.split(':')[0]
  name = '_'.join(name.split("/")[-2:])
  return name

def plot_learning_curves(records):
  """
  records : dict
      tensor_name -> {task1_name: [(epoch0_mean, epoch0_std), (epoch1_mean, epoch1_std), ...],
                      task2_name: [], ...}
  """
  n_records = len(records)
  import matplotlib
  from matplotlib import pyplot as plt

  V.plot_figure(nrow=4 * n_records, ncol=12)
  for tensor_idx, (tensor_name, records) in enumerate(
      sorted(records.items())):
    ax = V.plot_subplot(n_records, 1, tensor_idx + 1)
    handles = []

    for task_idx, (task_name, values) in enumerate(
        sorted(records.items())):
      mean = np.array([v[0] for v in values])
      std = np.array([v[1] for v in values])
      min_idx = np.argmin(mean)
      min_val = mean[min_idx]
      indices = np.arange(len(values), dtype='int32')

      _, = ax.plot(indices, [v[0] for v in values],
                   linestyle=linestyles[task_idx % len(linestyles)],
                   label=task_name)
      handles.append(_)
      c = _.get_color()

      ax.fill_between(indices, mean - std, mean + std,
                      zorder=0, color=c, alpha=0.2, linewidth=0.0)
      _ = matplotlib.patches.Patch(label="$\\sigma_{%s}$" % task_name,
                                   alpha=0.2, color=c)
      handles.append(_)

      _, = ax.plot(min_idx, min_val, color=c, alpha=0.6,
                   marker='o', markersize=10,
                   linestyle='None', linewidth=0.0,
                   label="$min_{%s}$" % task_name)
      handles.append(_)

    ax.set_xlabel("Epoch Count")
    plt.grid()
    plt.tight_layout()
    plt.legend(handles=handles, loc='best', markerscale=1.2, fontsize=12)
    plt.title(tensor_name)

# ===========================================================================
# Callbacks
# ===========================================================================
class LearningCurves(Callback):
  """ Some accessible properties from `odin.training.Task`:
   - curr_epoch: Total number of epoch finished since the beginning of the Task
   - curr_iter: Total number of iteration finished since the beginning of the Task
   - curr_samples: Total number of samples finished since the beginning of the Task
   - curr_epoch_iter: Number of iteration within current epoch
   - curr_epoch_samples: Number of samples within current epoch
  """

  def __init__(self, save_path):
    super(LearningCurves, self).__init__()
    assert os.path.isdir(save_path)
    self._save_path = str(save_path)
    # Mapping:
    # tensor_name -> {task1_name: [],
    #                 task2_name: [], ...}
    self._records = defaultdict(lambda: defaultdict(list))
    self._saved_epoch = 0

  def save_epoch(self, epoch):
    plot_learning_curves(self._records)
    V.plot_save(path=os.path.join(self._save_path, 'summary.png'),
                log=True, dpi=180)

  def epoch_end(self, task, epoch_results):
    task_name = task.name
    curr_epoch = task.curr_epoch

    for tensor_name, tensor_values in epoch_results.items():
      tensor_name = _normalize_name(tensor_name)
      if not isinstance(tensor_values[0], Number):
        continue
      self._records[tensor_name][task_name].append(
          (np.mean(tensor_values), np.std(tensor_values)))
    # ====== check if saving current epoch ====== #
    if all(len(j) > 3
           for i in self._records.values()
           for j in i.values()) and curr_epoch % 50 == 0:
      if self._saved_epoch == curr_epoch:
        return
      self._saved_epoch = curr_epoch
      self.save_epoch(curr_epoch)

  def task_end(self, task, task_results):
    self.save_epoch(task.curr_epoch)

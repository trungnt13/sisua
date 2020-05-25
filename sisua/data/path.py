# add path in order of priority
# DEFAULT_BASE_DIR: default prefix path if no configuration is found
# DATA_DIR: path for storing preprocessed data
# DOWNLOAD_DIR: path for storing downloaded (original) data
# EXP_DIR: path for storing experiments results
import os
from os.path import expanduser

from odin.utils import get_script_path, select_path

DEFAULT_BASE_DIR = expanduser("~")

if 'SISUA_DATA' in os.environ:
  DATA_DIR = os.environ['SISUA_DATA']
  if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
  elif os.path.isfile(DATA_DIR):
    raise RuntimeError("Store data path at '%s' must be a folder" % DATA_DIR)
else:
  DATA_DIR = select_path(os.path.join(DEFAULT_BASE_DIR, 'bio_data'),
                         create_new=True)

DOWNLOAD_DIR = select_path(os.path.join(DATA_DIR, 'downloads'), create_new=True)

# PATH for saving experiments results
if 'SISUA_EXP' in os.environ:
  EXP_DIR = os.environ['SISUA_EXP']
  if not os.path.exists(EXP_DIR):
    os.mkdir(EXP_DIR)
  elif os.path.isfile(EXP_DIR):
    raise RuntimeError("Experiment path at '%s' must be a folder" % EXP_DIR)
else:
  EXP_DIR = select_path(os.path.join(DEFAULT_BASE_DIR, 'bio_exp'),
                        create_new=True)

# ====== path for yaml configurations ====== #
if 'SISUA_CFG' in os.environ:
  CONFIG_PATH = os.path.abspath(os.environ['SISUA_CFG'])
else:
  CONFIG_PATH = os.path.abspath(
      os.path.join(get_script_path(__name__, return_dir=True), '..', '..',
                   'configs', 'base.yaml'))
if not os.path.isfile(CONFIG_PATH):
  raise RuntimeError("Cannot find configuration .yaml files at: %s" %
                     CONFIG_PATH)

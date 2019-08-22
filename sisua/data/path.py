# add path in order of priority
# DEFAULT_BASE_DIR: default prefix path if no configuration is found
# DATA_DIR/PREPROCESSED_BASE_DIR: path for storing preprocessed data
# DOWNLOAD_DIR: path for storing downloaded (original) data
# EXP_DIR: path for storing experiments results
import os
from os.path import expanduser

from odin.utils import select_path

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

PREPROCESSED_BASE_DIR = DATA_DIR
DOWNLOAD_DIR = select_path(os.path.join(DATA_DIR, 'downloads'), create_new=True)

# PATH for saving experiments results
if 'SISUA_EXP' in os.environ:
  EXP_DIR = os.environ['SISUA_EXP']
  if not os.path.exists(EXP_DIR):
    os.mkdir(EXP_DIR)
  elif os.path.isfile(EXP_DIR):
    raise RuntimeError("Experiment path at '%s' must be a folder" % EXP_DIR)
else:
  EXP_DIR = select_path(os.path.join(DEFAULT_BASE_DIR, 'bio_log'),
                        create_new=True)

# add path in order of priority
import os
from os.path import expanduser
from odin.utils import select_path

if 'SISUA_DATA' in os.environ:
  DATA_DIR = os.environ['SISUA_DATA']
  if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
  elif os.path.isfile(DATA_DIR):
    raise RuntimeError("Store data path at '%s' must be a folder" % DATA_DIR)
else:
  DATA_DIR = select_path(
      os.path.join(expanduser("~"), 'bio_data'),
      create_new=True)

PREPROCESSED_BASE_DIR = DATA_DIR
DOWNLOAD_DIR = select_path(
    os.path.join(DATA_DIR, 'downloads'),
    create_new=True)

# PATH for saving experiments results
if 'SISUA_EXP' in os.environ:
  EXP_DIR = os.environ['SISUA_EXP']
  if not os.path.exists(EXP_DIR):
    os.mkdir(EXP_DIR)
  elif os.path.isfile(EXP_DIR):
    raise RuntimeError("Experiment path at '%s' must be a folder" % EXP_DIR)
else:
  EXP_DIR = select_path(
      os.path.join(expanduser("~"), 'bio_log'),
      create_new=True)

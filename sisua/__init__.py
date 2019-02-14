from sisua.data.path import EXP_DIR, DATA_DIR, DOWNLOAD_DIR

_VERBOSE = False

def set_verbose(is_verbose_on):
  global _VERBOSE
  _VERBOSE = bool(is_verbose_on)

def is_verbose():
  return bool(_VERBOSE)

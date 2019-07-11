__version__ = '0.3.0'
_VERBOSE = False

def set_verbose(is_verbose_on):
  global _VERBOSE
  _VERBOSE = bool(is_verbose_on)

def is_verbose():
  return bool(_VERBOSE)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

# ===========================================================================
# Helper
# ===========================================================================
def get_tensorflow_version():
  import subprocess
  try:
    task = subprocess.Popen(["nvcc", "--version"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out = task.stdout.read()
    if "release 9.0" in str(out, 'utf-8'):
      return "tensorflow-gpu==1.12.0"
  except FileNotFoundError as e:
    pass
  return "tensorflow==1.12.0"

# ===========================================================================
# Main
# ===========================================================================
with open('README.md') as readme_file:
  readme = readme_file.read()

author = 'University of Eastern Finland'

requirements = [
    "odin-ai @ git+https://github.com/imito/odin-ai@0.1.2#egg=odin-0.1.2",
    get_tensorflow_version(),
    "tensorflow-probability==0.5.0",
    "seaborn>=0.9",
    "pandas",
]

setup(
    author=author,
    author_email='trung@imito.ai',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    description="SemI-SUpervised generative Autoencoder for single cell data",
    scripts=['bin/sisua-train'],
    setup_requires=['pip>=19.0'],
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='sisua',
    name='sisua',
    packages=find_packages(),
    test_suite='tests',
    url='https://github.com/trungnt13/sisua',
    version='0.1.0',
    zip_safe=False,
)

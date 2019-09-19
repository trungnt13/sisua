#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pip
from setuptools import setup, find_packages

_SISUA_VERSION = '0.4.2'

if not (sys.version_info.major == 3 and sys.version_info.minor == 6):
  raise RuntimeError("Sorry, we only support Python=3.6!")

if float(pip.__version__.split('.')[0]) < 19.0:
  raise RuntimeError(
      "'sisua' package require pip version >= 19.0, your pip version is %s, "
      "run `pip install pip --upgrade` to upgrade!" % str(pip.__version__))

# ===========================================================================
# Main
# ===========================================================================
with open('README.rst') as readme_file:
  readme = readme_file.read()

author = 'University of Eastern Finland'

requirements = [
    "odin-ai==1.1.1",
    "seaborn>=0.9",
    "pandas",
    'scanpy==1.4.4',
]

setup(
    author=author,
    author_email='trung@imito.ai',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    description="SemI-SUpervised generative Autoencoder for single cell data",
    long_description=readme,
    long_description_content_type='text/x-rst',
    scripts=['bin/sisua-train',
             'bin/sisua-analyze'],
    setup_requires=['pip>=19.0'],
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='sisua',
    name='sisua',
    packages=find_packages(),
    test_suite='tests',
    url='https://github.com/trungnt13/sisua',
    version=_SISUA_VERSION,
    zip_safe=False,
)

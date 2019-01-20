#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = [
    "numpy>=1.0, <1.15",
    "torch>=0.4.1",
    "matplotlib>=2.0",
    "scikit-learn>=0.18, <0.20.0",
    "scipy>=1.1",
    "h5py>=2.8",
    "pandas>=0.2",
    "loompy>=2.0",
    "tqdm >= 4",
    "anndata >= 0.6",
    "xlrd >= 1.0",
    "jupyter>=1.0.0",
    "nbconvert>=5.4.0",
    "nbformat>=4.4.0",
    "ipython>=7",
]

author = 'University of Eastern Finland'

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

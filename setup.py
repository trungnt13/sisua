#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

author = 'University of Eastern Finland'

requirements = [
    "odin-ai @ git+https://github.com/imito/odin-ai@0.1.2#egg=odin-0.1.2",
    "tensorflow-gpu",
    "tensorflow-probability",
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

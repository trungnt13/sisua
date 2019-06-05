SISUA
=====

|SISUA_design|

.. |SISUA_design| image:: https://drive.google.com/uc?export=view&id=1PvvG61_Rgbv_rqT6sCeb1XB6CtdiCMXX
  :width: 405
  :height: 249


Semi-supervised Single-cell modeling:

* Free software: MIT license
* Documentation: https://github.com/trungnt13/sisua/tree/master/docs.

Reference:

* Trung Ngo Trong, Roger Kramer, Juha Mehtonen, Gerardo González, Ville Hautamäki, Merja Heinäniemi. **"SISUA: SemI-SUpervised Generative Autoencoder for Single Cell Data"**, ICML Workshop on Computational Biology, 2019. `[pdf]`__

.. __: https://doi.org/10.1101/631382


Installation
************

You only need ``Python 3.6``, the stable version of SISUA installed via pip:

  ``pip install sisua``

Install the nightly version on github:

  ``pip install git+https://github.com/trungnt13/sisua@master``

For developers, we create a conda environment for SISUA contribution `sisua_env`__

  ``conda env create -f=sisua_env.yml``

.. __: https://github.com/trungnt13/sisua/blob/master/sisua_env.yml

Getting started
***************

a. The basics:
    * `Datasets description`__
    * `Models specification`
    * `Basic API and work-flow`__
b. Single-cell analysis:
    * `Latent space`
    * `Imputation of genes expression`
    * `Prediction of protein markers`
c. Advanced technical topics:
    * `Probabilistic embedding`__
    * `Hierarchical modeling` (*coming soon*)
    * `Causal analysis` (*coming soon*)
    * `Cross datasets analysis` (*coming soon*)
d. Benchmarks:
    * `Scalability test`__
    * `Fine-tuning networks`
    * `Data normalization`
e. Further development:
    * `Roadmap`__
    * `SISUA 2`__

.. __: https://github.com/trungnt13/sisua/blob/master/docs/dataset_description.md
.. __: https://github.com/trungnt13/sisua/blob/master/tutorials/basics.py
.. __: https://github.com/trungnt13/sisua/blob/master/tutorials/probabilistic_embedding.py
.. __: https://github.com/trungnt13/sisua/blob/master/tests/scalability.py
.. __:
.. __:

Toolkits
********

We provide binary toolkits for *fast and efficient* analyzing single-cell datasets:

* `sisua-train`__: train single-cell modeling algorithms, support training multiple systems in parallel.
* `sisua-analyze`__: evaluate, compare, and interpret trained model.
* `sisua-embed`__: probabilistic embedding for semi-supervised training.
* `sisua-data`__: *coming soon*


.. __: https://github.com/trungnt13/sisua/blob/master/bin/README.rst
.. __: https://github.com/trungnt13/sisua/blob/master/bin/README.rst
.. __: https://github.com/trungnt13/sisua/blob/master/bin/README.rst
.. __: https://github.com/trungnt13/sisua/blob/master/bin/README.rst

Some important arguments:

-model
            name of function declared in models__

            - ``scvi``: single-cell Variational Inference model
            - ``dca``: Deep Count Autoencoder
            - ``vae``: single-cell Variational Autoencoder
            - ``movae``: SISUA
-ds
            name of dataset declared in data__.

            Description of all predefined datasets is in docs__.

            Some good datasets for practicing:

            - ``pbmc8k_ly``
            - ``cortex``
            - ``pbmcecc_ly``
            - ``pbmcscvi``
            - ``pbmcscvae``

.. __: https://github.com/trungnt13/sisua/tree/master/sisua/models
.. __: https://github.com/trungnt13/sisua/tree/master/sisua/data
.. __: https://github.com/trungnt13/sisua/blob/master/docs/dataset_description.md

Configuration
*************

By default, the data will be saved at your home folder at ``~/bio_data``,
and the experiments' outputs will be stored at ``~/bio_log``

You can customize these two paths using the environment variables:

* For storing downloaded and preprocessed data: ``SISUA_DATA``
* For the experiments: ``SISUA_EXP``

For example:

.. code-block:: python

  import os
  os.environ['SISUA_DATA'] = '/tmp/bio_data'
  os.environ['SISUA_EXP'] = '/tmp/bio_log'

  from sisua.data import EXP_DIR, DATA_DIR

  print(DATA_DIR) # /tmp/bio_data
  print(EXP_DIR)  # /tmp/bio_log

or you could set the variables in advance:

.. code-block:: bash

  export SISUA_DATA=/tmp/bio_data
  export SISUA_EXP=/tmp/bio_log
  python sisua/train.py
  # or using the provided toolkit: sisua-train


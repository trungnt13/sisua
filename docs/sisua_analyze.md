# SISUA

SemI - SUpervised generative Autoencoder models for single cell data

<img src = "https://drive.google.com/uc?export=view&id=1PvvG61_Rgbv_rqT6sCeb1XB6CtdiCMXX"
  width = "584" height = "359" >

#### References

Trung Ngo Trong, Roger Kramer, Juha Mehtonen, Gerardo González, Ville Hautamäki, Merja Heinäniemi
**"SISUA: Semi-Supervised Generative Autoencoder for Single Cell Data" ** Submitted, 2019. [pdf](https://doi.org/10.1101/631382)

---
### Install

Install stable version of _SISUA_ using pip:

> pip install sisua

Install the nightly version on github:

> pip install git+https://github.com/trungnt13/sisua@master

For developers, we create a conda environment for SISUA contribution [sisua_env](https://github.com/trungnt13/sisua/blob/master/environment.yml)

> conda env create -f=sisua_env.yml

### Getting started

### SISUA toolkits

We provide a comprehensive list of toolkits for _fast & efficient_ single cell analysis.
These toolkits enable users getting started even without reading the API or dealing
with python code. Most important arguments for each toolkit are list in the following table,
for more information call `-h` option.

* `sisua-train`: training single-cell modeling algorithms. We support a wide coverage of
all state-of-the-art single-cell modeling algorithms `-model [scvi|dca|vae|movae|...]`, and
more thant **30** different dataset for reproducibility and comparison among different methods
`-ds [cortex|pbmcscvi|pbmcscvae|pbmc8k_ly|pbmcecc_ly|...]` (more information about each dataset
can be found at [metadata]()). Training multiple models in parallel also supported, for example,
`-ds cortex -model scvi,dca,vae,movae -nprocess 2` will run concurrently two configurations at once
and significantly speed up the process.
* `sisua-analyze`: quick and easy way to evaluate, compare and biologically interpret trained model,
you need to specify the name of the dataset as `first positional argument`,
then the name of all the models for the comparison `-model scvi,dca,vae`
* `sisua-data`: _coming soon_
* `sisua-embed`: _coming soon_


### Configuration

By default, the data will be saved at your home folder
at `~/bio_data`, and the experiments' outputs will be stored
at `~/bio_log`

You can customize these two paths using environment variable:

* For the data storing path: `SISUA_DATA`
* For the experiments path: `SISUA_EXP`

For example:

```python
import os
os.environ['SISUA_DATA'] = '/tmp/bio_data'
os.environ['SISUA_EXP'] = '/tmp/bio_log'

from sisua import EXP_DIR, DATA_DIR

print(DATA_DIR) # /tmp/bio_data
print(EXP_DIR)  # /tmp/bio_log
```

or you could set the variables in advance

```bash
export SISUA_DATA=/tmp/bio_data
export SISUA_EXP=/tmp/bio_log
python sisua/train.py
```


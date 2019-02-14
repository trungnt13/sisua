# sisua
SemI - SUpervised generative Autoencoder models for single cell data

### Requirement

* Python 3.6
* pip >= 19.0
* [odin - ai](https: // github.com / imito / odin - ai)

### Quick Start

You could install the package using pip:

`pip install git + https: // github.com / trungnt13 / sisua`

or using[conda](https: // conda.io / en / latest / miniconda.html) package manager and
our _environment.yml_ file:

`conda env create - f = environment.yml`

### Configure SISUA

By default, the downloaded data will be saved at your home folder
at path `~ / bio_data`, and the experiments' outputs will be stored
at path `~ / bio_log`

You can customize these two path using environment variable:

* For the data storing path: using variable `SISUA_DATA`
* For the experiments path: using variable `SISUA_EXP`

For example:

```python
import os
os.environ['SISUA_DATA'] = '/tmp/bio_data'
os.environ['SISUA_EXP'] = '/tmp/bio_log'

from sisua import EXP_DIR, DATA_DIR

print(DATA_DIR)
print(EXP_DIR)
```

### System description

![sisua](https://drive.google.com/uc?export=view&id=1CoyPcOTxa3mTYoeHH0t__AIq7p0rERe_)

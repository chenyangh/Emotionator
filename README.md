[<img src="img/pytorch-logo-dark.png" width="10%">](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


Suggested steps:

1. Create a new virtual environment, and install pytorch 1.0.1 with cuda

2. copy paste and run the follows:
```bash
pip install -r requirements.txt
git clone https://github.com/chenyangh/torchMoji.git
cd torchMoji
pip install -e .
python scripts/download_weights.py
```

For now it is only using CBET (subsets of CBET), I will make an update to support some of the other datasets **SOON**
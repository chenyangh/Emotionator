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

3. download and untar this [file](https://drive.google.com/file/d/1_G_nJkWKdsr-LO-uKRcTeT0Wf_uxG0_l/view?usp=sharing) and then move the files under the same dir of the trainers.
```bash
tar -zxvf pretrained.tar.gz
mv pretrained/* . 
```
For now it is only using CBET (subsets of CBET), I will make an update to support some of the other datasets **SOON**

4. run inference (type 'end' to stop)
```bash
python trainer_***.py -dataset=cbet/median/small
```


# Overview of the model used in lstm_elmo_deepmoji
[](img/overview.png)

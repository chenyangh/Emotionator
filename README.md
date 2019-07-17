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
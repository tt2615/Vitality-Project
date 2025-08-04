# Introduction
This code base consists of the source code of the deep learning BPR model and the baselines of the paper: Virality Analytics of Corporate Information Events: A Deep BPR
Framework with Attention Mechanisms submitted to ICAIF ’25.

# Data Samples
We disclose partial data for exhibition and verification purposes. Samples of training and testing data can be found under the folder data/, with 10,000 data each.

# How to use?
  1. Download the repo to local folder
  2. Run `pipenv install` to install dependencies
  3. Install PyTorch: e.g., for windows, run `pip3 install torch --index-url https://download.pytorch.org/whl/cu117`; for mac, run `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu`
  4. In the root directory, create new folders `data`, `logs`, `models`
  5. Open `data explore.ipynb`, select correct current kernel and run all cells to generate `processed_data.csv`
  6. Run `main.py` in the command line with appropriate parameters, for example:
    * train from scratch: ```python main.py --model=BPR_v2 --device=cuda --batch=1024 --lr=1e-3 --optim=Adam --epoch=50 --comment=log```
    * train from existing model: ```python main.py --model=BPR_v2 --device=cuda --batch=1024 --lr=1e-3 --optim=Adam --epoch=50 --comment=log --model_path=LR_1024_0.001_Adam_log```
    * test exisiting model: ```python main.py --device=cuda --mode=test --model_path=LR_1024_0.001_Adam_log```
     
# Machine Learning baselines
The machine learning baselines (XGboost and RF models) can be found in machine_learning_basedlines.ipynb. Ensure you have installed Jupyter and dependencies before running the code.

* How to use?
  1. download repo to local folder
  2. run `pipenv install` to install dependencies
  3. install pytorch: e.g., for windows, run `pip3 install torch --index-url https://download.pytorch.org/whl/cu117`; for mac, run `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu`
  4. in root directory, create new folders `data`, `logs`, `models`
  5. put `news_data_with_sentiment.csv` into `data` folder
  6. open `data explore.ipynb`, select correct current kernel and run all cells to generate `processed_data.csv`
  7. run `main.py` in command line with appropriate parameters, example:
    ```python main.py --model=LR --device=cuda --batch=1024 --lr=1e-3 --optim=Adam --epoch=50 --comment=log```
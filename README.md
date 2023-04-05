* How to use?
  1. download repo to local folder
  2. run `pipenv install` to install dependencies
  3. in root directory, create new models `data`, `logs`, `models`
  4. put `news_data_with_sentiment.csv` into `data` folder
  5. open `data explore.ipynb`, select correct kernel and run all cells to generate `processed_data.csv`
  6. run `main.py` in command line with appropriate parameters, example:
    ```python main.py --model=LR --device=cuda --batch=1024 --lr=1e-3 --optim=Adam --epoch=50 --comment=log```
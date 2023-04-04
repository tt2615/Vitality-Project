import pandas as pd
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset

class PostData(Dataset):
    def __init__(self, onehot_cols=[], tar_cols=[], x_transforms=None, y_transforms=None):

        #load data
        self.data = pd.read_csv(f"./data/processed_data.csv",index_col=0)

        #generate onehot encoding
        self.data = pd.get_dummies(self.data, columns=onehot_cols)

        #drop irrelevant columns
        text_cols = ['Item_Title', 'title', 'news_text']
        cat_cols = ['Item_Author', 'Company_ID', 'sentiment']
        self.data = self.data.drop(text_cols + cat_cols, axis=1, errors='ignore')
        self.tar_cols = tar_cols
        
        # print(self.data.dtypes)

        self.x_trans_list = x_transforms
        self.y_trans_list = y_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        x = record[[x for x in self.data.columns if x not in self.tar_cols]] 
        # ['sentence_length', 'word_length', 'word_!', 'word_:', 'word_?', 'score', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
        # print([x for x in self.data.columns if x not in self.tar_cols])
        y = record[self.tar_cols] # view, like, comment

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                x = trsfm(x) 
        if self.y_trans_list:
            for trsfm in self.y_trans_list:
                y = trsfm(y)
        
        return x, y

    def get_feature_num(self):
        return len(self.data.columns) - len(self.tar_cols)
    
# Data transforms

# Converts a numpy array to a torch tensor
class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data).float()
    

# Log the value
class Log(object):
    def __call__(self, data):
        return torch.clip(torch.log10(data), min=0.).float()
    
# Normalize
class Normalize(object):
    def __call__(self, data):
        return 


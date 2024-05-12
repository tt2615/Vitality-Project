import pandas as pd
import numpy as np

import torch
torch.manual_seed(666)
from torch.utils.data import Dataset

from transformers import BertTokenizer

class BprData(Dataset):
    def __init__(self, cat_cols=[], num_cols=[], topic_cols=[], user_cols=[], tar_col='viral', max_padding_len=32, dir="./data/eastmoney_bpr.csv", x_transforms=None, bert='bert-base-chinese'):

        #load data
        self.data = pd.read_csv(dir,sep='\\')
        # print(self.data.dtypes)

        ##---for pos cols-----
        self.cat_cols = cat_cols
        self.user_cols = user_cols
        self.num_cols = num_cols
        self.tar_col = tar_col
        self.topic_cols = topic_cols

        # process text data: for bert input 
        tokenizer = BertTokenizer.from_pretrained(bert)
        input_ids = []
        attention_masks = []
        for text in self.data['item_title']:
            encoded_dict = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                max_length=max_padding_len,
                                                truncation=True,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_tensors='pt')
            input_ids.append(np.array(encoded_dict['input_ids'].squeeze()))
            attention_masks.append(np.array(encoded_dict['attention_mask'].squeeze()))
            # print(text)
            # print(encoded_dict['input_ids'])
        self.data['title_id'] = input_ids
        self.data['title_mask'] = attention_masks

        self.text_cols=['title_id', 'title_mask']

        ##---for neg cols------

        # process text data: for bert input 
        neg_input_ids = []
        neg_attention_masks = []
        for text in self.data['item_title']:
            encoded_dict = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                max_length=max_padding_len,
                                                truncation=True,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_tensors='pt')
            neg_input_ids.append(np.array(encoded_dict['input_ids'].squeeze()))
            neg_attention_masks.append(np.array(encoded_dict['attention_mask'].squeeze()))
            # print(text)
            # print(encoded_dict['input_ids'])
        self.data['neg_title_id'] = input_ids
        self.data['neg_title_mask'] = attention_masks
        self.neg_text_cols=['neg_title_id', 'neg_title_mask']

        self.neg_cat_cols = ['neg_'+x for x in cat_cols]    
        self.neg_num_cols = ['neg_'+x for x in num_cols]
        self.neg_user_cols = ['neg_'+x for x in user_cols]
        self.neg_topic_cols = ['neg_'+x for x in topic_cols]

        self.x_trans_list = x_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]

        #----for pos columns:
        pos_text_input = record[self.text_cols]
        pos_non_text_input = record[self.num_cols+self.cat_cols+self.topic_cols].astype(np.float32)
        pos_user_input = record[self.user_cols]
        y = record[self.tar_col]
        # print(pos_non_text_input)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                pos_text_input = np.stack(pos_text_input.values)
                pos_text_input = trsfm(pos_text_input)
                pos_non_text_input = trsfm(pos_non_text_input)
                pos_user_input = trsfm(pos_user_input)
                y = trsfm(y)

        #----for neg columns:
        neg_text_input = record[self.neg_text_cols]
        neg_non_text_input = record[self.neg_num_cols+self.neg_cat_cols+self.neg_topic_cols].astype(np.float32)
        neg_user_input = record[self.neg_user_cols]
        # print(pos_non_text_input)
        # print(neg_user_input)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                neg_text_input = np.stack(neg_text_input.values)
                neg_text_input = trsfm(neg_text_input)
                neg_non_text_input = trsfm(neg_non_text_input)
                neg_user_input = trsfm(neg_user_input)
        
        return (pos_text_input, pos_non_text_input, pos_user_input, y), (neg_text_input, neg_non_text_input, neg_user_input)
    
    def get_task_num(self):
        return len(self.tar_cols)
    
    def get_cat_feature_unique_count(self):
        return [max(self.data[f'{x}'].max(),self.data[f'neg_{x}'].max())+1  for x in self.cat_cols]
    
    def get_num_feature_count(self):
        return len(self.num_cols)
    
    def get_cat_feature_count(self):
        return len(self.cat_cols)
    
    def get_user_feature_count(self):
        return len(self.user_cols)

    def get_topic_num(self):
        return len(self.topic_cols)
    
    def get_user_feature_unique_count(self):
        return [max(self.data[f'{x}'].max(),self.data[f'neg_{x}'].max())+1  for x in self.user_cols]
    
    def get_pos_data(self):
        pos_data = self.data[self.data['viral']==1]
        return pos_data
    
    def get_class_count(self):
        return self.data['viral'].value_counts()
    
    
import pandas as pd
import numpy as np
from os.path import exists
from tqdm import tqdm
from torch.utils.data import random_split

import torch
torch.manual_seed(666)
from torch.utils.data import Dataset

from transformers import BertTokenizer

class IncBprData(Dataset):
    def __init__(self, 
                 data_dir,
                 meta_dir,
                 mode,
                 post_cols=[], 
                 author_cols=[], 
                 tar_col='viral', 
                 max_padding_len=32, 
                 x_transforms=None, 
                 bert='bert-base-chinese'):
        
        
        self.post_cols = post_cols
        self.author_cols = author_cols
        self.tar_col = tar_col
        self.mode = mode
        self.x_trans_list = x_transforms

        self.data = pd.read_csv(data_dir, delimiter='<')

        # process text data: for bert input 
        tokenizer = BertTokenizer.from_pretrained(bert)
        input_ids = []
        attention_masks = []
        print("encode test data titles")
        for text in tqdm(self.data['item_title'], total=self.data.shape[0]):
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

        neg_input_ids = []
        neg_attention_masks = []
        print("encode neg data titles")
        for text in tqdm(self.data['neg_item_title'], total=self.data.shape[0]):
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
        self.data['neg_title_id'] = neg_input_ids
        self.data['neg_title_mask'] = neg_attention_masks

    
    # def get_post_feature_unique_count(self):
    #     return [self.data[x].nunique()+1 for x in self.post_cols]
    
    # def get_author_feature_unique_count(self):
    #     return [self.data[x].nunique()+1 for x in self.author_cols]
    
    def get_author_feature_count(self):
        return len(self.author_cols)

    def get_post_feature_count(self):
        return len(self.post_cols)

    # def get_pos_data(self):
    #     pos_data = self.data[self.data['viral']==1]
    #     return pos_data
    
    # def get_class_count(self):
    #     return self.data['viral'].value_counts()    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        #----for pos columns:
        pos_text_input = record[self.text_cols]
        pos_post_input = record[self.post_cols].astype(np.float32)
        pos_author_input = record[self.author_cols].astype(np.float32)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                pos_text_input = np.stack(pos_text_input.values)
                pos_text_input = trsfm(pos_text_input)
                pos_post_input = trsfm(pos_post_input)
                pos_author_input = trsfm(pos_author_input)

        #----for neg columns:
        neg_text_input = record[['neg_' + x for x in self.text_cols]]
        neg_post_input = record[['neg_' + x for x in self.post_cols]].astype(np.float32)
        neg_author_input = record[['neg_' + x for x in self.author_cols]].astype(np.float32)
        # print(pos_non_text_input)
        # print(neg_author_input)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                neg_text_input = np.stack(neg_text_input.values)
                neg_text_input = trsfm(neg_text_input)
                neg_post_input = trsfm(neg_post_input)
                neg_author_input = trsfm(neg_author_input)
        
        return (pos_text_input, pos_post_input, pos_author_input), (neg_text_input, neg_post_input, neg_author_input)



class IncTestData(Dataset):
    def __init__(self, 
                 data_dir,
                 meta_dir,
                 mode,
                 post_cols=[], 
                 author_cols=[], 
                 tar_col='viral', 
                 max_padding_len=32, 
                 x_transforms=None, 
                 bert='bert-base-chinese'):
        
        self.post_cols = post_cols
        self.author_cols = author_cols
        self.tar_col = tar_col
        self.mode = mode
        self.x_trans_list = x_transforms

        self.data = pd.read_csv(data_dir)
        self.metadata = (meta_dir)

        # process text data: for bert input 
        tokenizer = BertTokenizer.from_pretrained(bert)
        input_ids = []
        attention_masks = []
        print("encode test data titles")
        for text in tqdm(self.data['item_title'], total=self.data.shape[0]):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]

        text_input = record[self.text_cols]
        post_input = record[self.post_cols].astype(np.float32)
        author_input = record[self.author_cols].astype(np.float32)
        y = record[self.tar_col]

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                text_input = np.stack(text_input.values)
                text_input = trsfm(text_input)
                post_input = trsfm(post_input)
                author_input = trsfm(author_input)
                y = trsfm(y)

        return text_input, post_input, author_input, y
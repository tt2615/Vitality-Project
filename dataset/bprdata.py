import pandas as pd
import numpy as np

import torch
torch.manual_seed(666)
from torch.utils.data import Dataset

from transformers import BertTokenizer

class BprData(Dataset):
    def __init__(self, cat_cols=[], num_cols=[], tar_cols=[], user_cols=[], max_padding_len=32, dir="./data/eastmoney_topic_bpr.csv", x_transforms=None, bert='bert-base-chinese'):

        #load data
        self.data = pd.read_csv(dir,index_col=0, nrows=64000)
        # print(self.data.dtypes)

        # process cat cols: generate embed index for embed cols
        self.data[cat_cols] = self.data[cat_cols].apply(lambda c: c.astype('category'))
        self.embed_cols = []
        for cat_col in cat_cols:
            self.data[f'{cat_col}_index'] = self.data[cat_col].cat.codes+1 #+1 for nan cases
            self.embed_cols.append(f'{cat_col}_index')

        self.user_cols = []
        for user_col in user_cols:
            self.data[f'{user_col}_index'] = self.data[user_col].cat.codes+1 #+1 for nan cases
            self.user_cols.append(f'{user_col}_index')

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
            
        self.num_cols = num_cols
        self.tar_cols = tar_cols

        self.x_trans_list = x_transforms
        # self.y_trans_list = y_transforms

        # self.data = self.sample_pos_neg(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        text_input = record[self.text_cols]
        non_text_input = record[self.num_cols+self.embed_cols]
        user_input = record[self.user_cols]
        # print(non_text_input)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                text_input = np.stack(text_input.values)
                text_input = trsfm(text_input)
                non_text_input = trsfm(non_text_input)
                user_input = trsfm(non_text_input)
        
        return (text_input, non_text_input, user_input), (text_input, non_text_input, user_input)
    
    def sample_pos_neg(self, data, neg_sample_num):
        positive_rows = data[data['viral'] == 1]

        concatenated_rows = []
        neg_samples = 100
        null_count = 0

        # Iterate over each positive row
        for _, positive_row in positive_rows.iterrows():
            # Find corresponding negative row based on specified conditions
            negative_rows = data[(data['item_author_cate'] == positive_row['item_author_cate']) &
                                    (data['article_author'] == positive_row['article_author']) &
                                    (data['article_source_cate'] == positive_row['article_source_cate']) &
                                    (data['viral'] == 0)]
            
            # Check if there are valid negative rows
            if len(negative_rows)==0:
                negative_rows = data[(data['stock_code'] == positive_row['stock_code']) &
                                    (data['viral'] == 0)]
                
            if len(negative_rows)==0:
                null_count+=1


            elif 0<len(negative_rows)<=neg_sample_num:

                # Take the first negative row
                negative_row = negative_rows
            
            elif len(negative_rows)>neg_sample_num:
                # Take the first negative row
                negative_row = negative_rows.sample(n=neg_sample_num, replace=False)

            # Iterate over each sampled negative row
            for _, negative_row in negative_row.iterrows():
                # Concatenate negative row to positive row with modifications
                concatenated_row = pd.concat([positive_row, negative_row.add_prefix('neg_')])
                
                # Append concatenated row to the list
                concatenated_rows.append(concatenated_row)

        result_df = pd.DataFrame(concatenated_rows)
        return result_df
    
    def get_task_num(self):
        return len(self.tar_cols)
    
    def get_embed_feature_unique_count(self):
        return [self.data[x].nunique()+1 for x in self.embed_cols]
    
    def get_num_feature_count(self):
        return len(self.num_cols)
    
    def get_embed_feature_count(self):
        return len(self.embed_cols)
    
    def get_pos_data(self):
        pos_data = self.data[self.data['viral']==1]
        return pos_data
    
    def get_class_count(self):
        return self.data['viral'].value_counts()
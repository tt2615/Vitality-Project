import pandas as pd
import numpy as np
from os.path import exists
from tqdm import tqdm
from torch.utils.data import random_split

import torch
torch.manual_seed(666)
from torch.utils.data import Dataset

from transformers import BertTokenizer

class BprData():
    def __init__(self, cat_cols=[], num_cols=[], topic_cols=[], user_cols=[], tar_col='viral', dir="./data/eastmoney_bert.csv", max_padding_len=32, x_transforms=None, bert='bert-base-chinese'):

        #load data
        self.data = pd.read_csv(dir, nrows=64000)
        # print(self.data.dtypes)

        # # Drop the specified columns from the DataFrame
        # columns_to_exclude = []  # Add the names of the columns you want to exclude
        # self.data = self.data.drop(columns=columns_to_exclude)

        gen = torch.Generator()
        gen.manual_seed(666)
        train_data, valid_data, test_data = random_split(self.data, [0.8,0.1,0.1], generator=gen) #train:valid:test = 8:1:1

        self.cat_cols = cat_cols
        self.user_cols = user_cols
        self.num_cols = num_cols
        self.topic_cols = topic_cols
        self.tar_col = tar_col
        
        self.cat_cols_index = []
        self.data[cat_cols] = self.data[cat_cols].apply(lambda c: c.astype('category'))
        for cat_col in cat_cols:
            self.data[f'{cat_col}_index'] = self.data[cat_col].cat.codes+1 #+1 for nan cases
            self.cat_cols_index.append(f'{cat_col}_index')
        self.cat_cols = self.cat_cols_index

        self.user_cols_index = []
        self.data[user_cols] = self.data[user_cols].apply(lambda c: c.astype('category'))
        for user_col in user_cols:
            self.data[f'{user_col}_index'] = self.data[user_col].cat.codes+1 #+1 for nan cases
            self.user_cols_index.append(f'{user_col}_index')
        self.user_cols = self.user_cols_index

        train_dir = "./data/eastmoney_bpr_train.csv"
        valid_dir = "./data/eastmoney_bpr_valid.csv"
        self.train_data = BprSampledData(train_data.dataset.iloc[train_data.indices], 
                                         train_dir,
                                         self.cat_cols, 
                                         self.user_cols, 
                                         self.num_cols, 
                                         self.topic_cols,
                                         bert, 
                                         max_padding_len,
                                         x_transforms)
        self.valid_data = BprSampledData(valid_data.dataset.iloc[valid_data.indices],
                                         valid_dir,
                                         self.cat_cols,
                                         self.user_cols,
                                         self.num_cols,
                                         self.topic_cols,
                                         bert,
                                         max_padding_len,
                                         x_transforms)
        self.test_data = BprTestData(test_data.dataset.iloc[test_data.indices], 
                                     self.cat_cols, 
                                     self.user_cols, 
                                     self.num_cols, 
                                     self.topic_cols, 
                                     self.tar_col, 
                                     bert, 
                                     max_padding_len,
                                     x_transforms)
    
    def get_cat_feature_unique_count(self):
        return [self.data[x].nunique()+1 for x in self.cat_cols]
    
    def get_user_feature_unique_count(self):
        return [self.data[x].nunique()+1 for x in self.user_cols]
    
    def get_num_feature_count(self):
        return len(self.num_cols)
    
    def get_cat_feature_count(self):
        return len(self.cat_cols)
    
    def get_user_feature_count(self):
        return len(self.user_cols)

    def get_topic_num(self):
        return len(self.topic_cols)
    
    def get_pos_data(self):
        pos_data = self.data[self.data['viral']==1]
        return pos_data
    
    def get_class_count(self):
        return self.data['viral'].value_counts()
    

class BprSampledData(Dataset):
    def __init__(self, data, dir, cat_cols, user_cols, num_cols, topic_cols, bert, max_padding_len, x_transforms):
        
        if not exists(dir):
            self.form_bpr_train_data(data, dir)
        self.data = pd.read_csv(dir, delimiter='<')

        ##---for pos cols-----
        self.cat_cols = cat_cols
        self.user_cols = user_cols
        self.num_cols = num_cols
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
        pos_user_input = record[self.user_cols].astype(np.float32)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                pos_text_input = np.stack(pos_text_input.values)
                pos_text_input = trsfm(pos_text_input)
                pos_non_text_input = trsfm(pos_non_text_input)
                pos_user_input = trsfm(pos_user_input)

        #----for neg columns:
        neg_text_input = record[self.neg_text_cols]
        neg_non_text_input = record[self.neg_num_cols+self.neg_cat_cols+self.neg_topic_cols].astype(np.float32)
        neg_user_input = record[self.neg_user_cols].astype(np.float32)
        # print(pos_non_text_input)
        # print(neg_user_input)

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                neg_text_input = np.stack(neg_text_input.values)
                neg_text_input = trsfm(neg_text_input)
                neg_non_text_input = trsfm(neg_non_text_input)
                neg_user_input = trsfm(neg_user_input)
        
        return (pos_text_input, pos_non_text_input, pos_user_input), (neg_text_input, neg_non_text_input, neg_user_input)

    def form_bpr_train_data(self, data, dir):
        print(f"form bpr sampled data to :{dir}")
        positive_rows = data[data['viral'] == 1]

        neg_sample_num = 100
        null_count = 0

        # Open the output file
        with open(dir, 'w', encoding='UTF-8') as f:
            first_line = True

            # Iterate over each positive row
            for _, positive_row in tqdm(positive_rows.iterrows(), total=positive_rows.shape[0]):
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
                    continue

                elif len(negative_rows)==1:
                    neg_samples = negative_rows.iloc[0]
                    concatenated_row = pd.concat([positive_row, neg_samples.add_prefix('neg_')])
                    if first_line:
                        f.write('<'.join(map(str, concatenated_row.keys()))+'\n')
                        first_line = False

                    # Write the concatenated row to the file
                    f.write('<'.join(map(str, concatenated_row.values)) + '\n')
                    continue

                elif 1<len(negative_rows)<=neg_sample_num:
                    # Take the first negative row
                    neg_samples = negative_rows
                
                elif len(negative_rows)>neg_sample_num:
                    # Take all negative rows
                    neg_samples = negative_rows.sample(n=neg_sample_num, replace=False)
                
                # Iterate over each sampled negative row
                for _, negative_row in neg_samples.iterrows():
                    # Concatenate negative row to positive row with modifications
                    concatenated_row = pd.concat([positive_row, negative_row.add_prefix('neg_')])

                    if first_line:
                        f.write('<'.join(map(str, concatenated_row.keys()))+'\n')
                        first_line = False

                    # Write the concatenated row to the file
                    f.write('<'.join(map(str, concatenated_row.values)) + '\n')
                

class BprTestData(Dataset):
    def __init__(self, data, cat_cols, user_cols, num_cols, topic_cols, tar_col, bert, max_padding_len, x_transforms):
        self.data = data
        self.cat_cols = cat_cols
        self.user_cols = user_cols
        self.num_cols = num_cols
        self.topic_cols = topic_cols
        self.tar_col = tar_col

        #save test data for other model comparison
        dir = './data/eastmoney_bpr_test.csv'
        if not exists(dir):
            self.data.to_csv(dir, index=False)

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

        self.x_trans_list = x_transforms
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]

        #----for pos columns:
        text_input = record[self.text_cols]
        non_text_input = record[self.num_cols+self.cat_cols+self.topic_cols].astype(np.float32)
        user_input = record[self.user_cols].astype(np.float32)
        y = record[self.tar_col]

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                text_input = np.stack(text_input.values)
                text_input = trsfm(text_input)
                non_text_input = trsfm(non_text_input)
                user_input = trsfm(user_input)
                y = trsfm(y)

        return text_input, non_text_input, user_input, y
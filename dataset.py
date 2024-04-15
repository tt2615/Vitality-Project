import math
import pandas as pd

import torch
torch.manual_seed(0)
from torch.utils.data import Dataset

import math
import numpy as np
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

from transformers import BertTokenizer

class PostData(Dataset):
    def __init__(self, cat_cols=[], num_cols=[], tar_cols=[], max_padding_len=32, dir="./data/processed_data_wo_content.csv", x_transforms=None, y_transforms=None):

        #load data
        self.data = pd.read_csv(dir,index_col=0, nrows=64000) #,nrows=64
        # print(self.data.dtypes)

        # #register data
        # self.text_cols = ['item_title']
        # self.cat_cols = ['stock_code', 'item_author', 'article_author', 'article_source','eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism']
        # self.num_cols = ['item_views', 'item_comment_counts', 'article_likes']
        # self.embed_cols = ['stock_index', 'iauthor_index', 'aauthor_index', 'source_index']

        # #generate onehot encoding
        # self.data = pd.get_dummies(self.data, columns=onehot_cols)

        # process cat cols: generate embed index for embed cols
        self.data[cat_cols] = self.data[cat_cols].apply(lambda c: c.astype('category'))
        self.embed_cols = []
        for cat_col in cat_cols:
            self.data[f'{cat_col}_index'] = self.data[cat_col].cat.codes+1 #+1 for nan cases
            self.embed_cols.append(f'{cat_col}_index')

        # process text data: for bert input 
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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
        self.data['title_id'] = input_ids
        self.data['title_mask'] = attention_masks

        self.text_cols=['title_id', 'title_mask']
            
        self.num_cols = num_cols
        self.tar_cols = tar_cols

        # print(self.data.dtypes)

        self.x_trans_list = x_transforms
        self.y_trans_list = y_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        text_input = record[self.text_cols]
        non_text_input = record[self.num_cols+self.embed_cols]
        # print(non_text_input)
        y = record[self.tar_cols]

        if self.x_trans_list:
            for trsfm in self.x_trans_list:
                text_input = np.stack(text_input.values)
                text_input = trsfm(text_input)
                non_text_input = trsfm(non_text_input) 
        if self.y_trans_list:
            for trsfm in self.y_trans_list:
                y = trsfm(y)

        # print(text_input)
        # print(non_text_input)
        # print(y)
        
        return text_input, non_text_input, y
    
    def get_task_num(self):
        return len(self.tar_cols)
    
    def get_embed_feature_unique_count(self):
        return [self.data[x].nunique()+1 for x in self.embed_cols]
    
    def get_num_feature_count(self):
        return len(self.num_cols)
    
    def get_embed_feature_count(self):
        return len(self.embed_cols)
    
# Data transforms

# Converts a numpy array to a torch tensor
class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data)
    
# class TextInputToTensor(object):
#     def __call__(self, data, index):
#         return torch.from_numpy(data)
    

# Log the value
class Log(object):
    def __call__(self, data):
        return torch.clip(torch.log10(data), min=0.)
    
# Normalize
class Normalize(object):
    def __call__(self, data):
        return 

class TokenizeText(object):
    def __call__(self, data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        input_ids_title = []
        attention_masks_title = []
        for text in data['Item_Title']:
            encoded_dict = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                max_length=32,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
            input_ids_title.append(encoded_dict['input_ids'])
            attention_masks_title.append(encoded_dict['attention_mask'])
        input_ids_title = torch.cat(input_ids_title, dim=0)
        attention_masks_title = torch.cat(attention_masks_title, dim=0)

        input_ids_content = []
        attention_masks_content = []
        for text in data['news_text']:
            encoded_dict = tokenizer.encode_plus(text,
                                                add_special_tokens=True, 
                                                max_length=256,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
            input_ids_content.append(encoded_dict['input_ids'])
            attention_masks_content.append(encoded_dict['attention_mask'])
        input_ids_content = torch.cat(input_ids_content, dim=0)
        attention_masks_content = torch.cat(attention_masks_content, dim=0)

        return input_ids_title, attention_masks_title, input_ids_content, attention_masks_content



def random_split(dataset, lengths,
                 generator=torch.Generator().manual_seed(42)):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

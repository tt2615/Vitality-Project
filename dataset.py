import pandas as pd
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

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




def random_split(dataset, lengths,
                 generator=default_generator):
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

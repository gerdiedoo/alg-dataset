import torch 
import pandas as pd
import pytorch_lightning as pl

import io, tokenize, re, os

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

def get_datasets(df, tokenizer, split = 0.08, data_folder = "./data/prototype/"):
    x, y = df.shape
    test_size = int(x * 0.08)
    train_size = x - test_size    
    indices = torch.randperm(x)
    
    train_idx = indices[0:train_size]
    test_idx = indices[train_size:x]
    
    train = CodesDataset(df, train_idx, data_folder=data_folder, transform=VectorizeData(tokenizer))
    test = CodesDataset(df, test_idx, data_folder=data_folder, transform=VectorizeData(tokenizer))
    
    return train, test        

class CodesDataset(Dataset):
    def __init__(self, df, indices, data_folder = "./data/prototype/", transform=None):
        super(CodesDataset).__init__()
        
        assert isinstance(df, pd.DataFrame)
        
        self.df = df
        self.data_folder = data_folder
        (x, y) = df.shape
        self.indices = indices
        self.transform = transform
        
    def get_idx(self, idx):
        return self.indices[idx].item()
    
    def __len__(self):
        return self.indices.shape[0]
    
    def to_labels(self, row):
        # The input are of the form [1, 0, 1, 0, 0, 0, 1]
        # Ordering: Qsort, Msort, Ssort, Isort, Bsort, Lsearch, Bsearch, Llist, Hmap
        assert isinstance(row, pd.core.series.Series)
        
        qsort = int(row["Quicksort"])        # 1
        msort = int(row["Mergesort"])        # 2
        ssort = int(row["Selectionsort"])    # 3
        isort = int(row["Insertionsort"])    # 4
        bsort = int(row["Bubblesort"])       # 5
        lsearch = int(row["Linear search"])  # 6 
        bsearch = int(row["Binary Search"])  # 7
        llist = int(row["Linked List"])      # 8
        hmap  = int(row["Hashmap"])          # 9
        
        lst = [qsort, msort, ssort, isort, bsort, lsearch, bsearch, llist, hmap]
        
        return torch.tensor(lst)
    
    def __getitem__(self, idx):
        idx = self.get_idx(idx)
        row = self.df.iloc[idx]
        filename = row["Filename"]
        
        with open(self.data_folder + "/" + str(filename), "r") as f:
            file = f.read()
        labels = self.to_labels(row)
        
        if self.transform:
            sample = self.transform((file, labels))
        
        return sample
        
class VectorizeData(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        code, label = sample
        tokenizer = self.tokenizer
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        eos = tokenizer.eos_token
        pad = tokenizer.pad_token

        # Place the code into a form such that it will 
        # be tokenized by the CodeBERT tokenizer
        ss = list(map(lambda s : s.strip(), code.split("\n")))
        #print(ss)
        # Flatten the list. 
        #ss = [cls] + [item for sublist in ss for item in sublist]
        ss = cls + '\n'.join(ss)
        
        # Tokenize
        ss = tokenizer.encode_plus(ss, return_tensors='pt')

        # Pad
        _, x2 = ss['input_ids'].shape
        
        if x2 < 512:
            padding = torch.tensor([[1 for _ in range(512 - x2)]])

            ss['input_ids'] = torch.cat((ss['input_ids'], padding), dim=1).squeeze()
            padding = padding - 1
            ss['token_type_ids'] = torch.cat((ss['token_type_ids'], padding), dim=1).squeeze()
        else:
            ss['input_ids'] = ss['input_ids'][:, 0:512].squeeze()
            ss['token_type_ids'] = ss['token_type_ids'][:, 0:512].squeeze()

        return ss, label
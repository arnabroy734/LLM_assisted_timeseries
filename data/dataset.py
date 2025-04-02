import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import mean_squared_error
import pickle

def load_data(name, type):
    """
    Load dataset, scale those dataset, split (temporal 50:50) and return 
    Args:
        name(str): 'health' and 'wheat'
        type(str): 'time', 'text'
    """
    if name == 'wheat':
        data = pd.read_csv(Path.cwd()/'data/wheat_preprocessed.csv')
    elif name == 'health':
        data = pd.read_csv(Path.cwd()/'data/health_preprocessed.csv')
    if type == 'time':
        data = data[['target']]
        tr_size = int(data.shape[0]*0.5)
        train, test = data[:tr_size], data[tr_size:]
        scaler = StandardScaler()
        scaler.fit(train)
        train['target'] = scaler.transform(train)
        test['target'] = scaler.transform(test)
    elif type == 'text':
        data = data[['target', 'news']]
        tr_size = int(data.shape[0]*0.5)
        train, test = data[:tr_size], data[tr_size:]
        scaler = StandardScaler()
        scaler.fit(train.target.values.reshape((-1,1)))
        train['target'] = scaler.transform(train.target.values.reshape((-1,1)))
        test['target'] = scaler.transform(test.target.values.reshape((-1,1)))
        
    return train, test, scaler

class CustomDatasetLSTM(Dataset):
    def __init__(self, data, lag=7):
        """
        Args:
            data(pd.DataFrame): dim is (N , 1), only column is target
            lag(int): context or window length
        """
        self.X = list()
        self.y = list()
        for i in range(len(data) - lag):
            x = data.target.values[i : i + lag]
            y = data.target.values[i + lag]
            self.X.append(x.flatten())
            self.y.append(y)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   

class CustomDatasetText(Dataset):
    def __init__(self, data, tokeniser, lag=7):
        """
        Args:
            data(np.ndarray): dim is (N , 1)
            lag(int): context or window length
        Remarks:
            Max length of token is selected 512 with appropriate padding.
        """
        self.X = list()
        sentences = list()
        self.y = list()
        data = data.values
        for i in range(len(data) - lag):
            x = data[i : i + lag, 0]
            y = data[i + lag, 0]
            self.X.append(x.flatten())
            self.y.append(y)
            sentences.append(data[i+lag, 1])
        self.X = torch.tensor(np.array(self.X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y, dtype=np.float32), dtype=torch.float32)
        self.tokens = list()
        for text in sentences:
            ids = tokeniser.encode(text, max_length=512, truncation=True, padding='max_length')
            self.tokens.append(ids)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        mask_fn = lambda x: 1 if x!=0 else 0
        self.attn_masks = torch.clone(self.tokens)
        self.attn_masks.apply_(mask_fn)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.tokens[idx], self.attn_masks[idx], self.y[idx]   
    

def get_dataset(name, type, lag):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
    train, test, scaler = load_data(name, type)
    if type == 'time':
        trainds = CustomDatasetLSTM(train, lag)
        testds = CustomDatasetLSTM(test, lag)
    elif type == 'text':
        trainds = CustomDatasetText(train, tokenizer, lag)
        testds = CustomDatasetText(test, tokenizer, lag)
    return trainds, testds, scaler
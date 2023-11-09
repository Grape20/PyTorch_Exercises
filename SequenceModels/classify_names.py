# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:55:28 2023

@author: 85407
"""

#%% 
import os 

from clock import clock 

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

import torch 
from torch import nn 
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
from torch.utils.data import random_split 
from torch import optim  

import unicodedata
import string 
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            
#%% 

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def DataPreprosess(data_path): 
    category_lines = {} 
    all_categories = os.listdir(data_path)
    for idx, category in enumerate(all_categories): 
        filepath = os.path.join(data_path, category)
        category = category.split('.')[0]
        all_categories[idx] = category
        category_lines[category] = [unicodeToAscii(line) for line in open(filepath)] 
    
    return all_categories, category_lines 

#%% 

def letter2index(letter:str): 
    return all_letters.find(letter)

def line2tensor(line: str): 
    tensor = torch.zeros(len(line), n_letters) 
    for letter_id, letter in enumerate(line): 
        tensor[letter_id][letter2index(letter)] = 1 
    return tensor 

# line2tensor('Jones').size() 

#%% 
class LinesDataset(Dataset): 
    def __init__(self, all_categories, category_lines: dict): 
        self.data = category_lines  
        self.num_lines = [len(lines) for lines in self.data.values()] 
        self.all_categories = all_categories 
        
    def __len__(self):
        return sum(self.num_lines) 
    
    def __getitem__(self, index): 
        cum_num_lines = torch.cumsum(torch.tensor(self.num_lines), 0)
        category_id = -1 
        for cum_num_line in cum_num_lines: 
            category_id += 1 
            if index < cum_num_line: 
                break 
        if not category_id == 0: 
            index -= cum_num_lines[category_id-1]
        category = self.all_categories[category_id]
        try: 
            line = self.data[category][index] 
        except IndexError: 
            print(cum_num_lines) 
            print(index) 
            raise IndexError() 
        line_tensor = line2tensor(line)
        return category_id, line_tensor
    
#%% 
class RNNClassifyLines(nn.Module): 
    def __init__(self, num_categories: int, n_letters: int, hidden_size: int=128, *args, **kwargs): 
        super().__init__() 
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(n_letters, hidden_size, *args, **kwargs) 
        self.h2o = nn.Linear(hidden_size, num_categories, device=device)
        
    def forward(self, line, init_hidden=None): 
        if not init_hidden: 
            init_hidden = torch.zeros((1, line.shape[0], self.hidden_size), device=device)
        _, h_n = self.rnn(line, init_hidden) 
        logits = self.h2o(h_n[-1]) 
        return logits 
    
#%% 
@clock 
def train_loop(data_loader, model, loss_fn, optimizer): 
    size = len(data_loader.dataset)
    total_loss = 0 
    for batch, (y, X) in enumerate(data_loader): 
        X, y = X.to(device), y.to(device) 
              
        logits = model(X) 
        loss = loss_fn(logits, y) 
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        loss = loss.item() 
        
        if batch % 1000 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        total_loss += loss * len(X) 
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss 

#%% 
@clock 
def test_loop(data_loader, model, loss_fn): 
    test_loss = 0 
    
    with torch.no_grad(): 
        for batch, (y, X) in enumerate(data_loader): 
            X, y = X.to(device), y.to(device) 
            
            logits = model(X) 
            test_loss += loss_fn(logits, y).item() 
    
    test_loss /= len(data_loader) 
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss 

#%% 
def infer(model, line: str, all_categories): 
    logits = model(torch.unsqueeze(line2tensor(line), 0))
    logits = logits[0]
    probs = nn.functional.softmax(logits, 0)
    top5 = torch.topk(probs, 5) 
    print(f'RNN model infer name {line}: language(probability)')
    for idx in range(5): 
        print(f'{all_categories[top5[1][idx]]}({top5[0][idx]:.2f})')
    return all_categories[top5[1][0]]

#%%     
def plot_losses(losses): 
    plt.plot(np.array(losses).T) 

#%% 
def main(): 
    data_path = 'data/names'
    all_categories, category_lines = DataPreprosess(data_path)
    
    names_dataset = LinesDataset(all_categories, category_lines) 
    train_ratio = 0.7 
    train_dataset_size = int(len(names_dataset)*train_ratio)
    train_dataset, val_dataset = random_split(names_dataset, [train_dataset_size, len(names_dataset)-train_dataset_size])
    batch_size = 1 
    train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size), DataLoader(val_dataset, batch_size)
    
    RNN_classify_model = RNNClassifyLines(len(all_categories), n_letters, batch_first=True, device=device) 
    
    loss_fn = nn.CrossEntropyLoss()
    
    # optimizer = optim.SGD(RNN_classify_model.parameters(), lr=0.005)
    optimizer =  optim.Adam(RNN_classify_model.parameters())
    
    avg_losses = [[],[]] 
    num_epochs = 10 
    for epoch in range(num_epochs): 
        # 40s for one epoch training 
        train_avg_loss = train_loop(train_dataloader, RNN_classify_model, loss_fn, optimizer)
        avg_losses[0].append(train_avg_loss) 
        # 8s for one epoch validating 
        val_avg_loss = test_loop(val_dataloader, RNN_classify_model, loss_fn)
        avg_losses[1].append(val_avg_loss) 
    
    plot_losses(avg_losses)
    
    # TODO: save model and results 
        
if __name__ == '__main__': 
    main() 
    



# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:45:10 2023

@author: 85407
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 04:25:51 2022

@author: 85407
"""

#%% 
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T  

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#%% 
# transform = T.Compose([
#     T.Resize(256), 
#     # T.CenterCrop(224), 
#     T.RandomResizedCrop(224), 
#     T.RandomHorizontalFlip(), 
#     # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
#     # T.RandomPerspective(), 
#     # T.RandomRotation((0, 180)), 
#     # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET), 
    
#     T.ToTensor(), 
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
# ])
# # T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)

#%% 
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=T.ToTensor(), 
    # transform=transform, 
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=T.ToTensor(), 
    # transform=transform, 
)


train_val_dataloader = DataLoader(train_dataset, batch_size=64) 
train_dataset_ratio = 0.7 
train_dataset_size = int(len(train_dataset)*train_dataset_ratio)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_dataset_size, len(train_dataset)-train_dataset_size])

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

#%%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512, device=device),
            nn.ReLU(), 
            nn.Linear(512, 512, device=device),
            nn.ReLU(),
            nn.Linear(512, 10, device=device),
        )
        # weight scaling approximation, though Monte Carlo alternative
        self.dropout_linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(28*28, 512, device=device),
            nn.ReLU(),
            nn.Dropout(), 
            nn.Linear(512, 512, device=device),
            nn.ReLU(),
            nn.Dropout(), 
            nn.Linear(512, 10, device=device),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
# nn.Module.eval()
#%% 
class MultiTaskNetwork(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.flatten = nn.Flatten() 
        self.shared_stack = nn.Sequential(
            nn.Linear(224, 128, device=device), 
            nn.ReLU(), 
        )
        self.task1_stack = nn.Sequential(
            nn.Linear(128, 128, device=device), 
            nn.ReLU(), 
            nn.Linear(128, 10), 
        )
        self.task2_stack = nn.Sequential(
            nn.Linear(128, 128, device=device), 
            nn.ReLU(), 
            nn.Linear(128, 10), 
        )
        self.task3_stack = nn.Sequential(
            nn.Linear(128, 64, device=device), 
            nn.ReLU(), 
        )
    
    def forward(self, x): 
        x = self.flatten(x) 
        shared_hidden_units = self.shared_stack(x) 
        task1_logits = self.task1_stack(shared_hidden_units) 
        task2_logits = self.task2_stack(shared_hidden_units) 
        task3_features = self.task3_stack(shared_hidden_units)
        return [task1_logits, task2_logits, task3_features] 
    
# model = MultiTaskNetwork()
#%% 
learning_rate = 1e-4
batch_size = 64
epochs = 5

#%% 
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

#%% 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e6)
#%% 
# L2, L1 norm regularization
def parameter_norm_penalty(model, p, alpha): 
    if alpha == 0: return 0 
    weights = [] 
    for name, parameter in model.named_parameters(): 
        if 'weight' in name: 
            weights.append(torch.flatten(parameter))
    weights = torch.cat(weights)
    return alpha * torch.norm(weights, p)
#%% 
def tangent_prop_penalty(x: torch.Tensor, fx: torch.Tensor, tangent, alpha): 
    if alpha == 0: return 0 
    fx.backward(torch.ones_like(fx), retain_graph=True) 
    penalty = ((torch.matmul(tangent, x.grad.t()))**2).sum() 
    return alpha * penalty 
    
#%% 
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        
        # for tangent prop regularization
        X.requires_grad_(True)
        X = nn.Flatten()(X)
        X.retain_grad()
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # multitask learning
        # loss = 0.7 * loss_fn(pred[0], y) + (1-0.7) * loss_fn(pred[1, y])
        
        # L2, L1 norm regularization  
        p = 2
        # p = 1
        alpha = 0 
        loss += parameter_norm_penalty(model, p, alpha)
        
        # tangent prop regularization penalty 
        # notes: train slowly 
        alpha1 = 0 
        tangent = torch.ones((10, X.size(1)))
        loss += tangent_prop_penalty(X, pred, tangent, alpha1) 
        

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # # inject noise into weights
        # with torch.no_grad(): 
        #     for name, param in model.named_parameters(): 
        #         if 'weight' in name: 
        #             param += torch.randn(param.size())
            
        
        # # constrain parameters' norm penalty to be less than k  
        # k = 1 
        # for layer in model.linear_relu_stack: 
        #     if type(layer) == torch.nn.modules.linear.Linear: 
        #         layer._parameters['weight'] = nn.Parameter(torch.transpose(torch.transpose(layer._parameters['weight'],0,1)/torch.norm(layer._parameters['weight'], p=2, dim=1), 0, 1))
        # failed, and fuck pytorch and bing 
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    return loss 

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            p = 2
            # p = 1
            alpha = 0  
            test_loss += parameter_norm_penalty(model, p, alpha)
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(test_loss)
#%% 
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# def train(model, loss_fn, optimizer, epochs): 
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")  
# train(model, loss_fn, optimizer, epoch=5)

#%% 
correct_before, correct, t = 0, 0.0001, 0
while correct > correct_before: 
    correct_before = correct 
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    correct = test_loop(test_dataloader, model, loss_fn)
    t += 1 
print('Done!')

#%% 
def early_stopping(model: nn.Module, loss_fn, optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader, file_path='best_params.pth', period=1, patience=5, delta=0.01): 
    i, j = 0, 0
    v = np.Inf 
    torch.save(model.state_dict(), f=file_path)
    best_num_steps = i 
    while j < patience: 
        for _ in range(period): 
            train_loop(train_dataloader, model, loss_fn, optimizer)
        i += period 
        val_loss = test_loop(val_dataloader, model, loss_fn)
        if val_loss < v-delta: 
            best_num_steps = i 
            v = val_loss 
            torch.save(model.state_dict(), f=file_path)
            j = 0 
        else: 
            j += 1 
    print(f'Best parameters has been saved in {file_path}. \n')
    return best_num_steps, v 

file_path = 'best_params.pth' 
best_steps, min_val_loss = early_stopping(model, loss_fn, optimizer, train_dataloader, val_dataloader, file_path)

#%%
# strategy 1: Retraining on all data 
model = NeuralNetwork() 
# train(model, loss_fn, optimizer, epochs=best_steps) 
for t in range(best_steps): 
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_val_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")  

#%% 
# strategy 2: Continue Training on all data
max_num_epochs = 20 
for epoch in range(max_num_epochs): 
    print(f"Epoch {epoch+1+best_steps}\n-------------------------------")
    train_loop(train_val_dataloader, model, loss_fn, optimizer)
    val_loss = test_loop(val_dataloader, model, loss_fn)
    if val_loss < min_val_loss: break 
print("Done!")  


#%% 
class EarlyStopping(): 
    def __init__(self, file_path='best_params.pth', n=1, p=5, delta=0.01): 
        self.file_path = file_path 
        self.n = 1 
        self.p = 5 
        self.delta = delta
    
    def __call__(self, model: nn.Module, loss_fn, optimizer, train_dataloader: DataLoader, test_dataloader: DataLoader): 
        i, j = 0, 0
        v = np.Inf 
        torch.save(model.state_dict(), f=self.file_path)
        self.best_num_steps = i 
        while j < self.p: 
            for _ in range(self.n): 
                train_loop(train_dataloader, model, loss_fn, optimizer)
            i += self.n 
            val_loss = test_loop(val_dataloader, model, loss_fn)
            if val_loss < v-self.delta: 
                self.best_num_steps = i 
                v = val_loss 
                torch.save(model.state_dict(), f=self.file_path)
                j = 0 
            else: 
                j += 1 
        print(f'Best parameters has been saved in {self.file_path}. \n')
        return self.best_num_steps 




# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 04:25:51 2022

@author: 85407
"""

#%% 
import numpy as np 
from matplotlib import pyplot as plt 

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T 

from torchvision.models import resnet
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#%% 
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=T.ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=T.ToTensor()
)

train_val_dataloader = DataLoader(train_dataset, batch_size=64) 
# train_val_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # reshuffle the dataset to ensure the randomness 
# train_val_dataloader = DataLoader(train_dataset, batch_size=128) # for gradient-only methods 
# train_val_dataloader = DataLoader(train_dataset, batch_size=8192) # for second-order methods 
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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
# nn.Module.eval()
#%% 
class BatchNormNN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512, device=device, bias=False),
            nn.BatchNorm1d(512, momentum=0.1, device=device), 
            nn.ReLU(),
            nn.Linear(512, 512, device=device),
            nn.BatchNorm1d(512, device=device), 
            nn.ReLU(),
            nn.Linear(512, 10, device=device), 
            nn.BatchNorm1d(10, device=device), 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = BatchNormNN()
    
#%% 
learning_rate = 1e-4
batch_size = 64
epochs = 5

#%% 
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

#%% 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% 
def train_loop(dataloader, model, loss_fn, optimizer, grad_norm_trace):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # plot the norm of gradient over time 
        for index, parameter in enumerate(model.parameters()): 
            if len(grad_norm_trace) < index + 1: 
                grad_norm_trace.append([])
            grad_norm_trace[index].append(torch.norm(parameter.grad))

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    # plot the norm of gradient over time 
    return grad_norm_trace 


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(test_loss)
#%% 
epochs = 5
grad_norm_trace = [[]]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # plot the norm of gradient over time 
    train_loop(train_dataloader, model, loss_fn, optimizer, grad_norm_trace)
    # train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
plt.plot(np.array(grad_norm_trace).T)
#%% Experiment optimizers 
def experiment_optimizers(models:list, optimizers:list): 
    for index, model in enumerate(models): 
        epochs = 5 
        grad_norm_trace = [[]]
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            # plot the norm of gradient over time 
            train_loop(train_dataloader, model, loss_fn, optimizers[index], grad_norm_trace)
            # train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        plt.figure()
        plt.plot(np.array(grad_norm_trace).T)
    
#%% SGD
models = [NeuralNetwork() for _ in range(4)]
optimizers = [
    torch.optim.SGD(models[0].parameters(), lr=learning_rate, momentum=0.5), 
    torch.optim.SGD(models[1].parameters(), lr=learning_rate, momentum=0.9), 
    torch.optim.SGD(models[2].parameters(), lr=learning_rate, momentum=0.99), 
    torch.optim.SGD(models[3].parameters(), lr=learning_rate, momentum=0.9, nesterov=True), 
]

experiment_optimizers(models, optimizers)
#%% Adaptive Learning Rates
models1 = [NeuralNetwork() for _ in range(5)]
optimizers1 = [
    torch.optim.Adagrad(models1[0].parameters(), lr=1e-2), 
    torch.optim.RMSprop(models1[1].parameters(), lr=1e-2, alpha=0.99), 
    torch.optim.RMSprop(models1[2].parameters(), lr=1e-2, alpha=0.9), 
    torch.optim.RMSprop(models1[3].parameters(), lr=1e-2, momentum=0.5), 
    torch.optim.Adam(models1[4].parameters(), lr=1e-3, betas=(0.9, 0.999))
]

experiment_optimizers(models1, optimizers1)
#%% Second-Order methods 
models2 = [NeuralNetwork()]
optimizers2 = [torch.optim.LBFGS(models2[0].parameters())]

# step() missing 1 required positional argument: 'closure'
experiment_optimizers(models2, optimizers2)

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




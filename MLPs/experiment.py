# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 04:25:51 2022

@author: 85407
"""

#%% 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#%% 
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#%%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            
#             weight: the learnable weights of the module of shape
# (out\_features,in\_features). The values are initialized from U(−k,k), where k=1in\_features

# bias: the learnable bias of the module of shape (out\_features).
# If bias is True, the values are initialized from U(−k,k) where k=1in\_features
            nn.Linear(28*28, 512, device=device),
            nn.ReLU(),
            # nn.LeakyReLU(0.01), 
            # nn.LeakyReLU(-1), 
            # nn.PReLU(device=device), 
            # nn.MaxPool1d(kernel_size=k, stride=k, dilation=0, padding=0),  # maxout unit
            # nn.Sigmoid()
            # nn.Tanh()
            # nn.Identity(), 
            # nn.Softmax(), 
            # nn.Softplus(), # not recommended
            # nn.Hardtanh(), 
            nn.Linear(512, 512, device=device),
            nn.ReLU(),
            
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            # nn.Linear(512, 512, device=device),
            # nn.ReLU(),
            
            nn.Linear(512, 10, device=device),
            # nn.Sigmoid(), 
            # nn.logSoftmax(), 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
nn.Module.eval()
#%% 
learning_rate = 1e-4
batch_size = 64
epochs = 5

#%% 
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.L1Loss()
# loss_fn = nn.GaussianNLLLoss()
# loss_fn = nn.NLLLoss()

#%% 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e0, momentum=0.9)

#%% 
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # using GaussianNNLLoss
        # loss = nn.GaussianNLLLoss()(pred, nn.functional.one_hot(y), torch.ones(pred.shape)) # MSE Loss
        # loss = nn.GaussianNLLLoss()(pred[:, :8], nn.functional.one_hot(y)[:,:-2], nn.functional.relu(pred[:, -1])) # homoscedastic model
        # loss = nn.GaussianNLLLoss()(pred[:, :4], nn.functional.one_hot(y)[:,:4], nn.functional.relu(pred[:, -5])) # heteroscedastic model

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

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
loss_before, loss, t = 100, 100-0.0001, 0
while loss < loss_before: 
    loss_before = loss  
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss = test_loop(test_dataloader, model, loss_fn)
    t += 1 
print('Done!')


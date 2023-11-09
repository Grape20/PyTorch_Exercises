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
transform = T.Compose([
    # T.Resize(28), 
    # T.CenterCrop(224), 
    # T.RandomResizedCrop(224), 
    # T.RandomHorizontalFlip(), 
    # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
    # T.RandomPerspective(), 
    # T.RandomRotation((0, 180)), 
    # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET), 
    
    T.ToTensor(), 
    T.Normalize(mean=[0.5], std=[0.5]), 
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])
# T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)

#%% 
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform 
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform 
)

train_val_dataloader = DataLoader(train_dataset, batch_size=64) 
train_dataset_ratio = 0.7 
train_dataset_size = int(len(train_dataset)*train_dataset_ratio)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_dataset_size, len(train_dataset)-train_dataset_size])

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)
#%% 
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform 
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform 
)

train_val_dataloader = DataLoader(train_dataset, batch_size=64) 
train_dataset_ratio = 0.7 
train_dataset_size = int(len(train_dataset)*train_dataset_ratio)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_dataset_size, len(train_dataset)-train_dataset_size])

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


#%%
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
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

aMLP = MLP() 
# model = NeuralNetwork()
# nn.Module.eval()
#%%
class CNN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
        self.conv_layers = [] 
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=8, 
                kernel_size=3, 
                stride=2, 
                padding=0, # 'valid', default, range from 0 to 2 here
                # padding='same', 
                # padding_mode – 'zeros', 'reflect', 'replicate' or 'circular'
                # dilation=1, # default, actually no dilation in PyTorch 
                groups=1, # default, range from 1 to in_channels
                bias=True, # default 
                device=device
                ), 
            nn.BatchNorm2d(8, device=device), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                8, 32, 3, 2, device=device
                ), 
            nn.BatchNorm2d(32, device=device), 
            nn.ReLU(), 
            nn.MaxPool2d(3), 
            ))
        
        # self.conv_layers.append(nn.Sequential(
        #     nn.Conv2d(
        #         32, 64, 3, 2, device=device
        #         ), 
        #     nn.BatchNorm2d(64, device=device), 
        #     nn.ReLU(), 
        #     # nn.MaxPool2d(3), 
        #     ))
        
        # self.conv_layers.append(nn.Sequential(
        #     nn.Conv2d(
        #         64, 64, 3, 2, 
        #         ), 
        #     nn.BatchNorm2d(64), 
        #     nn.ReLU(), 
        #     nn.MaxPool2d(3), 
        #     ))
        
        self.flatten = nn.Flatten()
        
        self.fully_conect_layers = nn.Sequential(
            nn.Linear(32*6*6, 64, device=device), 
            nn.BatchNorm1d(64, device=device), 
            nn.ReLU(), 
            nn.Linear(64, 10, device=device), 
            nn.BatchNorm1d(10, device=device), 
            # nn.ReLU(), 
            # nn.Linear(32, 10, device=device), 
            # nn.BatchNorm1d(10, device=device), 
            )        
        
    def forward(self, x): 
        conv_output = x 
        for conv_layer in self.conv_layers: 
            conv_output = conv_layer(conv_output)
        logits = self.fully_conect_layers(self.flatten(conv_output))
        return logits 
    
#%%
class CNN1(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
        self.conv_layers = [] 
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=8, 
                kernel_size=3, 
                stride=2, 
                padding=1, # 'valid', default, range from 0 to 2 here
                # padding='same', 
                # padding_mode – 'zeros', 'reflect', 'replicate' or 'circular'
                # dilation=1, # default, actually no dilation in PyTorch 
                groups=1, # default, range from 1 to in_channels
                bias=True, # default 
                device=device
                ), 
            nn.BatchNorm2d(8, device=device), 
            nn.ReLU(), 
            # nn.MaxPool2d(kernel_size=3), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                8, 32, 3, 2, 1, device=device
                ), 
            nn.BatchNorm2d(32, device=device), 
            nn.ReLU(), 
            # nn.MaxPool2d(3), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                32, 64, 3, 2, device=device
                ), 
            nn.BatchNorm2d(64, device=device), 
            nn.ReLU(), 
            # nn.MaxPool2d(3), 
            ))
        
        # self.conv_layers.append(nn.Sequential(
        #     nn.Conv2d(
        #         64, 64, 3, 2, 
        #         ), 
        #     nn.BatchNorm2d(64), 
        #     nn.ReLU(), 
        #     nn.MaxPool2d(3), 
        #     ))
        
        self.flatten = nn.Flatten()
        
        self.fully_conect_layers = nn.Sequential(
            nn.Linear(64*3*3, 256, device=device), 
            nn.BatchNorm1d(256, device=device), 
            nn.ReLU(), 
            nn.Linear(256, 64, device=device), 
            nn.BatchNorm1d(64, device=device), 
            nn.ReLU(), 
            # nn.Linear(64, 64, device=device), 
            # nn.BatchNorm1d(64, device=device), 
            # nn.ReLU(), 
            nn.Linear(64, 10, device=device), 
            nn.BatchNorm1d(10, device=device), 
            # nn.ReLU(), 
            # nn.Linear(32, 10, device=device), 
            # nn.BatchNorm1d(10, device=device), 
            )        
        
    def forward(self, x): 
        conv_output = x 
        for conv_layer in self.conv_layers: 
            conv_output = conv_layer(conv_output)
        logits = self.fully_conect_layers(self.flatten(conv_output))
        return logits 
    
#%% 
myCNN = CNN() 
myCNN1 = CNN1() 
        
#%% 
class CSDNCNN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
        self.conv_layers = [] 
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=2, 
                padding=1, # 'valid', default, range from 0 to 2 here
                # padding='same', 
                # padding_mode – 'zeros', 'reflect', 'replicate' or 'circular'
                # dilation=1, # default, actually no dilation in PyTorch 
                groups=1, # default, range from 1 to in_channels
                bias=True, # default 
                device=device
                ), 
            nn.BatchNorm2d(16, device=device), 
            nn.ReLU(), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                16, 32, 3, 2, 1, device=device
                ), 
            nn.BatchNorm2d(32, device=device), 
            nn.ReLU(), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                32, 64, 3, 2, 1, device=device
                ), 
            nn.BatchNorm2d(64, device=device), 
            nn.ReLU(), 
            # nn.MaxPool2d(3), 
            ))
        
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(
                64, 64, 2, 2, 
                ), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            ))
        
        self.flatten = nn.Flatten()
        
        self.fully_conect_layers = nn.Sequential(
            nn.Linear(64*2*2, 100, device=device), 
            # nn.BatchNorm1d(64, device=device), 
            # nn.ReLU(), 
            nn.Linear(100, 10, device=device), 
            # nn.BatchNorm1d(10, device=device), 
            # nn.ReLU(), 
            # nn.Linear(32, 10, device=device), 
            # nn.BatchNorm1d(10, device=device), 
            )        
        
    def forward(self, x): 
        conv_output = x 
        for conv_layer in self.conv_layers: 
            conv_output = conv_layer(conv_output)
        logits = self.fully_conect_layers(self.flatten(conv_output))
           
        return logits 
    
aCSDNCNN = CSDNCNN() 

#%% 
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
    
class UnsharedConvNetwork(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
        self.layer1 = nn.Sequential(
            LocallyConnected2d(1, 8, 13, 3, 2), 
            nn.ReLU(), 
            )
        # self.layer1 = LocallyConnected2d(1, 8, 27, 2, 1)
        self.flatten = nn.Flatten() 
        self.fully_connected_layer = nn.Linear(8*13*13, 10)
        # self.fully_connected_layer = nn.Linear(8*27*27, 10)
        
    def forward(self, x): 
        output = self.fully_connected_layer(self.flatten(self.layer1(x))) 
        return output 

a_unshared_conv_network = UnsharedConvNetwork() 

#%% 
import torchvision.models.alexnet
import torchvision.models.googlenet
import torchvision.models.inception
import torchvision.models.resnet
import torchvision.models.vgg

#%% 
# learning_rate = 1e-4
# batch_size = 64
# epochs = 5

#%% 
# Initialize the loss function
# loss_fn = nn.CrossEntropyLoss()

#%% 
# optimizer = torch.optim.SGD(yCNN.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(myCNN.parameters(), momentum=0.5), 
# optimizer = torch.optim.Adam(myCNN.parameters())

#%% 
def train_loop(dataloader, model, loss_fn, optimizer):
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

def pre_exper(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

#%% 
pre_exper(
    train_dataloader, 
    test_dataloader, 
    # model=aCSDNCNN,  
    # model=myCNN1, 
    model=aMLP, 
    # model = a_unshared_conv_network, 
    # model=myCNN, 
    loss_fn=nn.CrossEntropyLoss(), 
    # optimizer=torch.optim.Adam(aCSDNCNN.parameters()), 
    # optimizer=torch.optim.Adam(myCNN1.parameters()), 
    optimizer=torch.optim.Adam(aMLP.parameters()), 
    # optimizer=torch.optim.Adam(a_unshared_conv_network.parameters()), 
    # optimizer=torch.optim.SGD(aCSDNCNN.parameters(), 1e-3, 0.99), 
    # optimizer=torch.optim.RMSprop(aCSDNCNN.parameters()), 
    )

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
#%%
file_path = 'best_params.pth' 
best_steps, min_val_loss = early_stopping(
    # model=aCSDNCNN,  
    model=myCNN1, 
    loss_fn=nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(aCSDNCNN.parameters()), 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    file_path='best_params.pth', 
    )

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
    train_loop(train_val_dataloader, aCSDNCNN, nn.CrossEntropyLoss(), torch.optim.Adam(aCSDNCNN.parameters()))
    val_loss = test_loop(val_dataloader, aCSDNCNN, nn.CrossEntropyLoss())
    if val_loss < min_val_loss: break 
print("Done!")  




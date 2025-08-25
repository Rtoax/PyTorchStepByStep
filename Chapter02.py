#!/bin/env python

from config import *
config_chapter2()
# This is needed to render the plots in this chapter
from plots.chapter2 import *

import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

#matplotlib inline
plt.style.use('fivethirtyeight')


true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them
# into PyTorch's Tensors and then we send them to the
# chosen device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)

################################################################################

def test_1():
    # This is redundant now, but it won't be when we introduce
    # Datasets...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Computes model's predicted output - forward pass
        yhat = model(x_train_tensor)

        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y_train_tensor)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()

        # Step 4 - Updates parameters using gradients and
        # the learning rate
        optimizer.step()
        optimizer.zero_grad()

    print(model.state_dict())


################################################################################

def make_train_step_fn(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def perform_train_step_fn(x, y):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # Step 3 - Computes gradients for both "a" and "b" parameters
        loss.backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return perform_train_step_fn

def test_2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Creates the train_step function for our model, loss function and optimizer
    train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

    # Defines number of epochs
    n_epochs = 1000

    losses = []

    # For each epoch...
    for epoch in range(n_epochs):
        # Performs one train step and returns the corresponding loss
        loss = train_step_fn(x_train_tensor, y_train_tensor)
        losses.append(loss)

    # Checks model's parameters
    print(model.state_dict())


################################################################################

def test_3():
    # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])

    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    next(iter(train_loader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Creates the train_step function for our model, loss function and optimizer
    train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

    # Defines number of epochs
    n_epochs = 1000

    losses = []

    # For each epoch...
    for epoch in range(n_epochs):
        # inner loop
        mini_batch_losses = []
        for x_batch, y_batch in train_loader:
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Performs one train step and returns the corresponding loss
            # for this mini-batch
            mini_batch_loss = train_step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        # Computes average loss over all mini-batches - that's the epoch loss
        loss = np.mean(mini_batch_losses)

        losses.append(loss)

    # Checks model's parameters
    print(model.state_dict())

################################################################################
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss

def test_4():
    # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # Builds Dataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)

    # Builds DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Creates the train_step function for our model, loss function and optimizer
    train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

    # Defines number of epochs
    n_epochs = 200

    losses = []

    for epoch in range(n_epochs):
        # inner loop
        loss = mini_batch(device, train_loader, train_step_fn)
        losses.append(loss)

    # Checks model's parameters
    print(model.state_dict())


################################################################################
def make_val_step_fn(model, loss_fn):
    # Builds function that performs a step in the validation loop
    def perform_val_step_fn(x, y):
        # Sets model to EVAL mode
        model.eval()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
        return loss.item()

    return perform_val_step_fn

def test_5():
    torch.manual_seed(13)

    # Builds tensors from numpy arrays BEFORE split
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # Builds dataset containing ALL data points
    dataset = TensorDataset(x_tensor, y_tensor)

    # Performs the split
    ratio = .8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train

    train_data, val_data = random_split(dataset, [n_train, n_val])

    # Builds a loader of each set
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=16)

    # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # Builds Dataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)

    # Builds DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Creates the train_step function for our model, loss function and optimizer
    train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

    # Creates the val_step function for our model and loss function
    val_step_fn = make_val_step_fn(model, loss_fn)

    # Defines number of epochs
    n_epochs = 200

    losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # inner loop
        loss = mini_batch(device, train_loader, train_step_fn)
        losses.append(loss)

        # VALIDATION
        # no gradients in validation!
        with torch.no_grad():
            val_loss = mini_batch(device, val_loader, val_step_fn)
            val_losses.append(val_loss)

    # Checks model's parameters
    print(model.state_dict())

    fig = plot_losses(losses, val_losses)
    plt.show()

test_5()
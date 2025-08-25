#!/bin/env python

from config import *
config_chapter1()
# This is needed to render the plots in this chapter
from plots.chapter1 import *

import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot


true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

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

figure1(x_train, y_train, x_val, y_val)
plt.show()

def my_LinearRegression_torch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Our data was in Numpy arrays, but we need to transform them
    # into PyTorch's Tensors and then we send them to the
    # chosen device
    x_train_tensor = torch.as_tensor(x_train).float().to(device)
    y_train_tensor = torch.as_tensor(y_train).float().to(device)

    # Here we can see the difference - notice that .type() is more
    # useful since it also tells us WHERE the tensor is (device)
    print(type(x_train), type(x_train_tensor), x_train_tensor.type())

    back_to_numpy = x_train_tensor.numpy()
    back_to_numpy = x_train_tensor.cpu().numpy()

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, \
                    dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, \
                    dtype=torch.float, device=device)

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train_tensor

        # Step 2 - Computes the loss
        # We are using ALL data points, so this is BATCH gradient
        # descent. How wrong is our model? That's the error!
        error = (yhat - y_train_tensor)
        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()

        # We can try plotting the graph for any python variable:
        # yhat, error, loss...
        make_dot(yhat)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        # No more manual computation of gradients!
        # b_grad = 2 * error.mean()
        # w_grad = 2 * (x_tensor * error).mean()
        # We just tell PyTorch to work its way BACKWARDS
        # from the specified loss!
        loss.backward()

        # Step 4 - Updates parameters using gradients and
        # the learning rate. But not so fast...
        # FIRST ATTEMPT - just using the same code as before
        # AttributeError: 'NoneType' object has no attribute 'zero_'
        # b = b - lr * b.grad
        # w = w - lr * w.grad
        # print(b)

        # SECOND ATTEMPT - using in-place Python assigment
        # RuntimeError: a leaf Variable that requires grad
        # has been used in an in-place operation.
        # b -= lr * b.grad
        # w -= lr * w.grad

        # THIRD ATTEMPT - NO_GRAD for the win!
        # We need to use NO_GRAD to keep the update out of
        # the gradient computation. Why is that? It boils
        # down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            b -= lr * b.grad
            w -= lr * w.grad

        # PyTorch is "clingy" to its computed gradients, we
        # need to tell it to let it go...
        b.grad.zero_()
        w.grad.zero_()

    print(b, w)


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "b" and "w" real parameters of the model,
        # we need to wrap them with nn.Parameter
        self.b = nn.Parameter(torch.randn(1,
                                          requires_grad=True,
                                          dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1,
                                          requires_grad=True,
                                          dtype=torch.float))

    def forward(self, x):
        # Computes the outputs / predictions
        return self.b + self.w * x

# Nested Models
class MyLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear model
        # with single input and single output
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # Now it only takes a call
        self.linear(x)


def my_LinearRegression_torch_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Our data was in Numpy arrays, but we need to transform them
    # into PyTorch's Tensors and then we send them to the
    # chosen device
    x_train_tensor = torch.as_tensor(x_train).float().to(device)
    y_train_tensor = torch.as_tensor(y_train).float().to(device)

    # Here we can see the difference - notice that .type() is more
    # useful since it also tells us WHERE the tensor is (device)
    print(type(x_train), type(x_train_tensor), x_train_tensor.type())

    back_to_numpy = x_train_tensor.numpy()
    back_to_numpy = x_train_tensor.cpu().numpy()

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


#my_LinearRegression_torch()
my_LinearRegression_torch_model()

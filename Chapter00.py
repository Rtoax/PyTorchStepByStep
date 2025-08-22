#!/bin/env python

from config import *
config_chapter0()
# This is needed to render the plots in this chapter
from plots.chapter0 import *

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import optim, nn


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

def my_LinearRegression():
    # Step 0 - Initializes parameters "b" and "w" randomly
    np.random.seed(42)
    b = np.random.randn(1)
    w = np.random.randn(1)

    print(b, w)

    figure2(x_train, y_train, b, w)
    plt.show()

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1
    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train

        # Step 2 - Computes the loss
        # We are using ALL data points, so this is BATCH gradient
        # descent. How wrong is our model? That's the error!
        error = (yhat - y_train)
        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()

        # Step 3 - Computes gradients for both "b" and "w" parameters
        b_grad = 2 * error.mean()
        w_grad = 2 * (x_train * error).mean()

        # Step 4 - Updates parameters using gradients and
        # the learning rate
        b = b - lr * b_grad
        w = w - lr * w_grad

        figure9(x_train, y_train, b, w)
        plt.show()

    print(b, w)


    # Sanity Check: do we get the same results as our
    # gradient descent?
    linr = LinearRegression()
    linr.fit(x_train, y_train)
    print(linr.intercept_, linr.coef_[0])

    fig = figure3(x_train, y_train, b, w)
    plt.show()

    # we have to split the ranges in 100 evenly spaced intervals each
    b_range = np.linspace(true_b - 3, true_b + 3, 101)
    w_range = np.linspace(true_w - 3, true_w + 3, 101)
    # meshgrid is a handy function that generates a grid of b and w
    # values for all combinations
    bs, ws = np.meshgrid(b_range, w_range)
    bs.shape, ws.shape

    sample_x = x_train[0]
    sample_yhat = bs + ws * sample_x
    sample_yhat.shape

    all_predictions = np.apply_along_axis(
        func1d=lambda x: bs + ws * x,
        axis=1,
        arr=x_train
    )
    all_predictions.shape

    all_labels = y_train.reshape(-1, 1, 1)
    all_labels.shape

    all_errors = (all_predictions - all_labels)
    all_errors.shape

    all_losses = (all_errors ** 2).mean(axis=0)
    all_losses.shape

    figure4(x_train, y_train, b, w, bs, ws, all_losses)
    plt.show()

    figure5(x_train, y_train, b, w, bs, ws, all_losses)
    plt.show()

    figure6(x_train, y_train, b, w, bs, ws, all_losses)
    plt.show()

    figure7(b, w, bs, ws, all_losses)
    plt.show()

    figure8(b, w, bs, ws, all_losses)
    plt.show()


my_LinearRegression()
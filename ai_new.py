"""
The artificially intelligent self driving car simulation brain.
Created by: Sultan Sidhu
Date: Sunday, December 23, 2018
*Created as a part of the Neural Networks and Artificial Intelligence course undertaken on Udemy.
"""

import numpy as np
import random
import os  # loading and saving features through this library
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim  # optimizer for stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import variable
from typing import Deque

# Creating Neural Network Architecture


class Network(nn.Module):
    """Creates the neural network for the self-driving car."""

    def __init__(self, input_size, nb_action):
        """Initializes the object of type Network."""
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        # the last two lines define how many neurons in each layer. in current setup, input_size in input layer,
        # 30 hidden neurons, and nb_action output neurons

    def forward(self, state):
        """Function that performs forward propogation, hence activating the neural network"""
        x = func.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Experience Replay Implementation


class Replay(object):
    """A class handling experience replay for the model."""

    def __init__(self, capacity=100):
        self.capacity = capacity  # the number of past experiences you want to store.
        self.memory = Deque(maxlen=capacity)  # Deque is a collection type that will store stuff up till a max length
        # and pop everything thereafter

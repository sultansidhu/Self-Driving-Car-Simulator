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
from torch.autograd import Variable
from typing import Deque

# Creating Neural Network Architecture


class Network(nn.Module):
    """Creates the neural network for the self-driving car."""

    def __init__(self, input_size, nb_action):
        """Initializes the object of type Network."""
        super(Network, self).__init__()
        self.input_size = input_size  # the number of dimensions in the vectors that are encoding the states
        self.nb_action = nb_action  # number of possible actions that the agent can make
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

    def __init__(self, capacity=100000):
        self.capacity = capacity  # the number of past experiences you want to store.
        self.memory = Deque(maxlen=capacity)  # Deque is a collection type that will store stuff up till a max length
        # and pop everything thereafter

    def sample(self, sample_size):
        """Gets a random sample from the memory of experiences."""
        samples = zip(*random.sample(self.memory, sample_size))
        # The zip function converts sublist format. So for all sublists within the given list, the zip function will
        # convert it to form sublists of all indices i of both lists together.
        # if there are more things in one sublist than are in the other sublist, the elements with no counterpart
        # in the shorter sublist are ignored by the function
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep-Q Learning

class DeepQNetwork:
    """A class representing the Deep Q-Learning Network for the self driving car."""

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = Replay()  # we took defualt = 100,000
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # optimizer performing stochastic gradient descent
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        """Selects the optimal action for the agent to take."""
        probabilities = func.softmax(self.model.forward(Variable(state, volatile=True))*7)  # Temperature parameter = 7
        # Temperature parameters change the probabilities by polarizing them
        # Higher temp param => higher probs get higher and lower probs get lower
        action = probabilities.multinomial()
        return action.data[0, 0]

    

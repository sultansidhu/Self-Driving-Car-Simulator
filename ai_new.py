"""
The artificially intelligent self driving car simulation brain.
Created by: Sultan Sidhu
Date: Sunday, December 23, 2018
*Created as a part of the Neural Networks and Artificial Intelligence course undertaken on Udemy.
"""
import random
import os  # loading and saving features through this library
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim  # optimizer for stochastic gradient descent
from torch.autograd import Variable
from collections import deque

# Creating Neural Network Architecture


class Network(nn.Module):
    """Creates the neural network for the self-driving car."""

    def __init__(self, input_size, nb_action):
        """Initializes the object of type Network."""
        super(Network, self).__init__()
        self.input_size = input_size  # the number of dimensions in the vectors that are encoding the states
        self.nb_action = nb_action  # number of possible actions that the agent can make
        self.fc1 = nn.Linear(input_size, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc2 = nn.Linear(60, nb_action)
        # the last two lines define how many neurons in each layer. in current setup, input_size in input layer,
        # 30 hidden neurons, and nb_action output neurons

    def forward(self, state):
        """Function that performs forward propogation, hence activating the neural network"""
        x = func.relu(self.fc1(state))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        q_values = self.fc2(x)
        return q_values

# Experience Replay Implementation


class Replay(object):
    """A class handling experience replay for the model."""

    def __init__(self, capacity=1000000):
        self.capacity = capacity  # the number of past experiences you want to store.
        self.memory = deque(maxlen=capacity)  # Deque is a collection type that will store stuff up till a max length
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
        self.reward_window = deque(maxlen=1000)
        self.model = Network(input_size, nb_action)
        self.memory = Replay()  # we took defualt = 100,000
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # optimizer performing stochastic gradient descent
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        """Selects the optimal action for the agent to take."""
        probabilities = func.softmax(self.model.forward(Variable(state, volatile=True))*100)
        # Temperature parameter = 7 but made = 0 in order to deactivate the AI
        # Temperature parameters change the probabilities by polarizing them
        # Higher temp param => higher probs get higher and lower probs get lower
        action = probabilities.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """Chungus"""
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  # because the action index is 1
        target = self.gamma * next_outputs + batch_reward
        td_loss = func.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()  # reinitializes the optimizer at each iteration
        td_loss.backward()
        self.optimizer.step()  # updates weights according to their respective contribution to the td_loss

    def update(self, reward, new_signal):
        """Updates the neural network per iteration / move made."""
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.memory.append((
            self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 1000:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        return action

    def score(self):
        """Calculates score for the agent."""
        return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        """Saves the current state of the agent in its environment, along with its learning."""
        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()},
                   "last_brain.pth")

    def load(self):
        """Loads a previously saved agent and brain."""
        if os.path.isfile("last_brain.pth"):
            print("Loading saved brain...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done!')
        else:
            print('No checkpoint found! Start new learning session!')

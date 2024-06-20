from objs import Car, Environment
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# Defining the feed-forward neural network model

class DQN(nn.Module):
    def __init__(self, in_states, h1, h2, out_actions):
        super().__init__
        self.first = nn.Linear(in_states, h1)
        self.a1 = F.relu()
        self.second = nn.Linear(h1, h2)
        self.a2 = F.relu()
        self.out = nn.Linear(h2, out_actions)

    def forward(self, x):
        x = self.a1(self.first(x))
        x = self.a2(self.second(x))
        x = self.out(x)
        return x


# Replay memory: Deque containing certain number of past transitions
# Transitions have information on (state, action, new_state, reward)

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# The training and testing deep Q-learning model

class ModelDQL:
    # Hyperparameters
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9
    REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 50     # size of training data sampled from replay memory

    # NN
    loss_function = nn.MSELoss()
    optimizer = None    # initialized later

    def train(self):
        agent = Car()
        env = Environment()

        # STATES should include information upto certain number of columns ahead and behind
        
        #       length number of cols ---- behind the agent postion
        #       2 * length number of cols ---- including and ahead the agent position

        # Also, 2 additional state nodes for y position and angle of the car

        NUM_STATES = (env.GRID_ROWS * agent.length * 3) + 2
        HIDDEN_LAYERS = [100, 50]
        MAX_ROTATION_PER_STEP = 30
        NUM_ROTATION_ACTIONS = 30

        memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        eps = 1     # epsilon initialized to 100% (select completely random actions initially)
        eps_history = []

        # creating policy NN and target NN (and then copying the weights and biases of policy into target)
        policyDQN = DQN(NUM_STATES, *HIDDEN_LAYERS, NUM_ROTATION_ACTIONS)
        targetDQN = DQN(NUM_STATES, *HIDDEN_LAYERS, NUM_ROTATION_ACTIONS)
        targetDQN.load_state_dict(policyDQN.state_dict())

        # using Adam optimizer
        self.optimizer = optim.Adam(policyDQN.parameters(), lr=self.LEARNING_RATE)


if __name__ == "__main__":
    model = ModelDQL()
    model.train()

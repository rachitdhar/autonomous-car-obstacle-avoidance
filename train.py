from objs import Car, Environment
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining the feed-forward neural network model

class DQN(nn.Module):
    def __init__(self, in_states, h1, h2, out_actions):
        super(DQN, self).__init__()
        self.first = nn.Linear(in_states, h1)
        self.second = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_actions)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = F.relu(self.second(x))
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
    agent = Car()
    env = Environment()
    
    # Hyperparameters
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9
    REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 50             # size of training data sampled from replay memory
    MIN_MEMORY_EXPERIENCE = 30  # min size of memory after which optmization can be performed
    TRUNCATION_STEPS = 1000      # number of steps after which to stop the training/testing for that epoch

    # NN
    loss_function = nn.MSELoss()
    optimizer = None    # initialized later
    nn_sync_rate = 10   # number of steps the policy NN takes before syncing to target NN
    
    REWARDS = {
        'normal_step': 1,
        'collision': -100,
        'no_x_motion': -1,
        'backward_x_motion': -10,
        'turning': -0.1
    }

    # number of multiples of the car length that should be covered horizontally in the state
    RANGE_OF_VISIBILITY = 3

    # actions
    MAX_ROTATION_PER_STEP = 30
    NUM_ROTATION_ACTIONS = 30
    A_STEP = float(MAX_ROTATION_PER_STEP) / NUM_ROTATION_ACTIONS        # difference between two consecutive possible rotation actions
    ACTIONS = np.arange(-MAX_ROTATION_PER_STEP, MAX_ROTATION_PER_STEP, A_STEP)

    def train(self, epochs):

        # STATES should include information upto certain number of columns ahead and behind
        
        #       length number of cols ---- behind the self.agent postion
        #       2 * length number of cols ---- including and ahead the self.agent position

        # Also, 2 additional state nodes for y position and angle of the car

        num_states = (self.env.GRID_ROWS * self.agent.length * self.RANGE_OF_VISIBILITY) + 2
        HIDDEN_LAYERS = [100, 50]

        memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        eps = 1     # epsilon initialized to 100% (select completely random actions initially)
        eps_history = []

        # creating policy NN and target NN (and then copying the weights and biases of policy into target)
        policyDQN = DQN(num_states, *HIDDEN_LAYERS, self.NUM_ROTATION_ACTIONS)
        targetDQN = DQN(num_states, *HIDDEN_LAYERS, self.NUM_ROTATION_ACTIONS)
        targetDQN.load_state_dict(policyDQN.state_dict())

        # using Adam optimizer
        self.optimizer = optim.Adam(policyDQN.parameters(), lr=self.LEARNING_RATE)

        # track of rewards per epoch
        rewards_per_epoch = np.zeros(epochs)

        steps = 0

        for i in range(epochs):
            # initialize state
            state = self.getState(num_states)
            terminated = False
            truncated = False       # to stop the training after a certain number of steps
            reward = 0.0

            while (not terminated and reward < 10.0):
                # select action using epsilon-greedy method
                action = 0
                if random.random() < eps:
                    action = self.ACTIONS[random.randint(0, len(self.ACTIONS) - 1)]
                else:
                    with torch.no_grad():
                        action = policyDQN(self.getState(num_states, state)).argmax().item()

                # execute action and retrieve new state and reward
                new_state, reward_change, terminated = self.execute_action(action, num_states)
                reward += reward_change

                # save in memory
                memory.push((state, action, new_state, reward, terminated))

                # move to next state
                state = new_state
                steps += 1

                if (steps > self.TRUNCATION_STEPS):
                    truncated = True

            rewards_per_epoch[i] = reward
            #print(np.sum(rewards_per_epoch))
            if (len(memory) > self.MIN_MEMORY_EXPERIENCE and np.sum(rewards_per_epoch) > 0.0):
                batch = memory.sample(self.BATCH_SIZE)
                
                # optimze
                self.optimize(batch, policyDQN, targetDQN)

                # decay epsilon
                eps = max(0, eps - 1.0/epochs)
                eps_history.append(eps)

                # copy into target if min steps for sync achieved
                if (steps > self.nn_sync_rate):
                    targetDQN.load_state_dict(policyDQN.state_dict())
                    steps = 0
        
        torch.save(policyDQN.state_dict(), "dql_policy.pt")

        tot_reward = np.zeros(epochs)
        for i in range(epochs):
            tot_reward[i] = np.sum(rewards_per_epoch[max(0, i-100):(i + 1)])
        
        # plotting total reward history
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(tot_reward)
        
        # plotting epsilon history
        plt.subplot(1, 2, 2)
        plt.plot(eps_history)

        # save the plots
        plt.savefig("dql_plots.png")


    def execute_action(self, action, num_states):
        x_shift = self.agent.update_pos(action)
        self.env.shift(abs(x_shift))
        new_state = self.getState(num_states)

        if self.env.intersectsWith(self.agent):
            reward_change = self.REWARDS['collision']
            return [new_state, reward_change, True]
        elif abs(action) > 0.1:
            reward_change = self.REWARDS['turning']
            return [new_state, reward_change, False]
        elif abs(x_shift) < 0.01:
            reward_change = self.REWARDS['no_x_motion']
            return [new_state, reward_change, False]
        elif x_shift < 0.0:
            reward_change = self.REWARDS['backward_x_motion']
            return [new_state, reward_change, False]
        else:
            reward_change = self.REWARDS['normal_step']
            return [new_state, reward_change, False]


    def optimize(self, batch, policyDQN: DQN, targetDQN: DQN):
        # get number of input features
        num_states = policyDQN.first.in_features
        
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in batch:
            target = None
            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    val = reward + self.DISCOUNT_FACTOR * targetDQN(self.getState(num_states, new_state)).max()
                    target = torch.FloatTensor(val)

            # Get the current set of Q values
            current_q = policyDQN(self.getState(num_states, state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = targetDQN(self.getState(num_states, state)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole batch
        loss = self.loss_function(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def getState(self, num_states, state = None)->torch.Tensor:
        state_tensor = None
        car_col = self.agent.gridpos()[0]

        if (state is None):
            i = 0
            state_tensor = torch.zeros(num_states)
            for col in range(car_col - self.agent.length, car_col + self.agent.length * (self.RANGE_OF_VISIBILITY - 1)):
                for row in range(0, self.env.GRID_ROWS):
                    state_tensor[i] = self.env.grid[row][col]
                    i += 1
        else:
            state_tensor = torch.tensor(state)

        return state_tensor
    

    def test(self, epochs):
        num_states = (self.env.GRID_ROWS * self.agent.length * self.RANGE_OF_VISIBILITY) + 2
        HIDDEN_LAYERS = [100, 50]

        # loading the learner policy from .pt file
        policyDQN = DQN(num_states, *HIDDEN_LAYERS, self.NUM_ROTATION_ACTIONS)
        policyDQN.load_state_dict(torch.load("dql_policy.pt"))
        policyDQN.eval()    # to switch to evaluation mode
        
        steps = 0
        for i in range(epochs):
            state = self.getState(num_states)
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            action = 0
            while(not terminated and not truncated):  
                # Select best action
                with torch.no_grad():
                    action = policyDQN(self.getState(num_states, state)).argmax().item()

                # Execute action
                state, reward_change, terminated = self.execute_action(action, num_states)

                steps += 1
                if (steps > self.TRUNCATION_STEPS):
                    truncated = True


if __name__ == "__main__":
    model = ModelDQL()
    model.train(1000)
    model.test(10)

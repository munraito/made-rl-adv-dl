from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'next_state', 'reward']
)


class ReplayMemory():
    "Experince buffer / replay memory helper class for DQN"
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*exptuple)
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class QNetwork(nn.Module):
    "Classic DQN"
    def __init__(self, hid_size, game_size, action_size):
        """Initialize parameters and build model."""
        super(QNetwork, self).__init__()
        self.hid_size = hid_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=hid_size, kernel_size=game_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.fc2 = nn.Linear(hid_size, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv(state).view(-1, self.hid_size)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)
    

class DuelingDQNet(nn.Module):
    "Duieling DQN architecture"
    def __init__(self, hid_size, game_size, action_size):
        super(DuelingDQNet, self).__init__()
        self.hid_size = hid_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=hid_size, kernel_size=game_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.fc_adv = nn.Linear(hid_size, action_size)
        self.fc_val = nn.Linear(hid_size, 1)
        
    def forward(self, state):
        x = self.conv(state).view(-1, self.hid_size)
        x = self.fc1(x)
        x = F.relu(x)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return val + (adv - adv.mean())

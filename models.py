import torch  
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, seed, hidden1_size=400, hidden2_size=300):
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.linear1 = nn.Linear(obs_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, action_size)
        
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return F.tanh(self.linear3(out))


class Critic(nn.Module):
    def __init__(self, obs_size, action_size, seed, hidden1_size=400, hidden2_size=300):
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        self.linear1 = nn.Linear(obs_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, action_size)

    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        return self.linear3(out)

import numpy as np
import random

from models import Actor, Critic
from utils import *

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
EPSILON = 1.0
EPSILON_DECAY = 0.99
UPDATE_EVERY = 4       # how often to update the network
UPDATE_TIMES = 1       # how many time to learn for each update step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        policy_lr=LR,
        critic_lr=LR):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = EPSILON

        # Noise process
        self.noise = OUNoise(action_size, seed)
        
        random.seed(seed)

        # Networks
        self.policy_local = Actor(self.state_size, self.action_size, seed)
        self.policy_target = Actor(self.state_size, self.action_size, seed)
        self.critic_local = Critic(self.state_size + self.action_size, self.action_size, seed)
        self.critic_target = Critic(self.state_size + self.action_size, self.action_size, seed)

        # initialize target networks weights
        for target_param, param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(param.data)

        # optimizer
        self.policy_optimizer = optim.Adam(self.policy_local.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATE_TIMES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_local.eval()
        with torch.no_grad():
            action_values = self.policy_local(state).cpu().data.numpy()
        self.policy_local.train()
        if add_noise:
            action_values += self.epsilon * self.noise.sample()

        return np.clip(action_values, -1, 1)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # update critic
        Q_expected = self.critic_local(states, actions)
        next_actions = self.policy_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # print("CRITIC LOSS:", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # update actor
        predicted_actions = self.policy_local(states)
        policy_loss = -self.critic_local(states, predicted_actions).mean()

        # print("POLICY LOSS:", policy_loss)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.epsilon *= EPSILON_DECAY

        # ------------------- update target networks ------------------- #
        self.soft_update(self.policy_local, self.policy_target, TAU)   
        self.soft_update(self.critic_local, self.critic_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



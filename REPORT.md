[//]: # (Image References)

[version1]: ./scores_1.png "1 Agent"
[version2]: ./scores_20.png "20 Agents"

# Project 2: Continuous COntrol

# Project Implementation

## Algorithm

I solved this task using DDPG as the action space is continuous.

## Model

The actor and critic network models are defined in `model.Actor` and `model.Critic`,
with 2 fully-connected layers of (400, 300) neurons with RELU activation, followed by
another fully connected layer of `action_size` neurons.
For the actor networks the final layer usees tanh activation to limit output to (-1, 1).
For the critic networks, the final layer directly outputs action-values.

## Agent

The DDPG Agent is defined in `agent.DDPGAgent`. The agent handles the learning process
in `Agent.learn()`, where the update steps for the local actor and critic networks take place. 

`Agent.act()` handles the mapping from `states -> actions`. During learning,
Ornstein–Uhlenbeck process noise is added to the actions to encourage exploration.

## Hyperparameters

Following hyperparameters were used:
* replay buffer size: 1e5
* minibatch size: 128 
* discount factor: 0.99 
* target network soft update factor: 1e-3  
* learning rate: 1e-4 
* update every n steps: 20
* number of learning steps for each update step: 10
* starting epsilon: 1.0
* epsilon decay: 0.99

## Results

The first version with 1 agent was solve in ~300 episodes

![1 Agent][version1]

The second version with 20 agents was solve in ~40 episodes

![20 Agents][version2]

## Observation

My initial implementation using Ornstein–Uhlenbeck noise was failing to learn.
I then added an epsilon-greedy decay to the noise samples to improve learning.

## Future improvements

Some different things to try:
* adding Prioritized Experience Replay
* adapting A2C/A3C to use output continuous actions
* TRPO and PPO

Last but not least more rigorous hyperparamter tuning would provide better results.

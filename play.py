from unityagents import UnityEnvironment
import numpy as np

from agent import DDPGAgent

import torch


def play():
    env = UnityEnvironment(file_name='./Reacher.app')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    # create agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size, seed=0)

    # load weights
    agent.policy_local.load_state_dict(torch.load('policy.pth'))

    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    state = env_info.vector_observations[0]                # get the current state (for each agent)
    score = 0                                              # initialize the score (for each agent)
    while True:
        action = agent.act(state, add_noise=False)         # select an action (for each agent)
        env_info = env.step(action)[brain_name]            # send all actions to tne environment
        next_state = env_info.vector_observations[0]       # get next state (for each agent)
        reward = env_info.rewards[0]                       # get reward (for each agent)
        done = env_info.local_done[0]                      # see if episode finished
        score += reward                                    # update the score (for each agent)
        state = next_state                                 # roll over states to next time step
        if done:                                           # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(score))

    env.close()


if __name__ == "__main__":
    play()

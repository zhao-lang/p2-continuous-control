from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from agent import DDPGAgent

import torch


GOAL_SCORE = 30.0
SCORE_WINDOW = 100
LOG_EVERY = 1

def train(env, agent, n_episodes=150, max_t=1000):
    scores = []
    avg_scores = []
    scores_window = deque(maxlen=SCORE_WINDOW)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()

        states = env_info.vector_observations  
        tmp_scores = np.zeros(num_agents)

        for t in range(max_t):
            actions = agent.act(states, add_noise=True)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)

            states = next_states
            tmp_scores += rewards

            if np.any(dones):
                break 
        
        score = np.mean(tmp_scores)
        scores_window.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_window))
        
        # if np.mean(scores_window) >= GOAL_SCORE:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        #     print('SOLVED')
        #     break

        if i_episode % LOG_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    
    torch.save(agent.policy_local.state_dict(), 'policy.pth')
    torch.save(agent.critic_local.state_dict(), 'critic.pth')
                  
    return scores, avg_scores


if __name__ == "__main__":
    env = UnityEnvironment(file_name='./Reacher20.app')

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

    # do training
    scores, avg_scores = train(env, agent)

    env.close()

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='score')
    plt.plot(np.arange(len(avg_scores)), avg_scores, c='r', label='avg score')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.savefig('scores_20.png')

import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add
import matplotlib.pyplot as plt

from DQN_agents import DQN_Agent, Double_DQN_Agent, Dueling_DQN_Agent, Double_Dueling_DQN_Agent, Expected_SARSA_Agent

def train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size):
    counter = 0
    reward_list = []
    action_num = env.action_space.n
    one_hot_action = np.eye(action_num)

    for episode in range(num_episodes):
        done, total_reward = False, 0

        state, _ = env.reset()

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            agent.buffer.store(state, one_hot_action[action], reward, next_state, done)

            if agent.buffer.size() >= batch_size and counter%replay_period[0] == 0:
                if agent.memory_type == 'PER':
                    td, idxs = agent.replay_experience(batch_size)
                    for i in range(batch_size):
                        agent.buffer.update(idxs[i], abs(td[i]))
                else:
                    agent.replay_experience(batch_size)
            
            if agent.buffer.size() >= batch_size and counter%replay_period[1] == 0:
                agent.update_target()

            state = next_state
        
        if agent.memory_type == 'PER':
            if episode >= beta_anneal_episodes[0]:
                agent.buffer.beta = agent.buffer.anneal_beta(episode, beta_anneal_episodes[0], num_episodes[1], 0.4, 1)
            if episode > beta_anneal_episodes[1]:
                agent.buffer.beta = 1

        reward_list.append(total_reward)
        print(f'Episode: {episode+1}, total reward: {total_reward}')

    agent_name = agent.__class__.__name__.split('_')[:-1]
    if not os.path.exists(agent_name):
        os.makedirs(agent_name)

    agent.save_model(os.path.join(agent_name))

    return reward_list

## DQN
env = gym.make('CartPole-v1')
agent = DQN_Agent(input_shape=env.observation_space.shape, 
                  num_actions=env.action_space.n,
                  gamma=0.99,
                  epsilon=1,
                  epsilon_decay=0.999,
                  epsilon_min=0.05,
                  memory_type='',
                  buffer_size=2**15,
                  pretrained='')
num_episodes = 1000
beta_anneal_episodes = (200, 800)
replay_period = (1,100)
batch_size = 128

reward_list = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size)

fig_title = 'DQN'
x_axis = np.arange(len(reward_list))+1
plt.figure(figsize=[3,3], dpi=300)
plt.title(fig_title, fontsize=9)
plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
plt.xlable('Episodes', fontsize=7)
plt.ylabel('Total Rewards', fontsize=7)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(linewidth=.1)
plt.savefig('figs/'+fig_title+'.png', bbox_inches='tight')
plt.close()


## Double DQN
env = gym.make('CartPole-v1')
agent = Double_DQN_Agent(input_shape=env.observation_space.shape, 
                         num_actions=env.action_space.n,
                         gamma=0.99,
                         epsilon=1,
                         epsilon_decay=0.999,
                         epsilon_min=0.05,
                         memory_type='',
                         buffer_size=2**15,
                         pretrained='')
num_episodes = 1000
beta_anneal_episodes = (200, 800)
replay_period = (1,100)
batch_size = 128

reward_list = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size)

fig_title = 'Double_DQN'
x_axis = np.arange(len(reward_list))+1
plt.figure(figsize=[3,3], dpi=300)
plt.title(fig_title, fontsize=9)
plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
plt.xlable('Episodes', fontsize=7)
plt.ylabel('Total Rewards', fontsize=7)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(linewidth=.1)
plt.savefig('figs/'+fig_title+'.png', bbox_inches='tight')
plt.close()


## Dueling DQN
env = gym.make('CartPole-v1')
agent = Dueling_DQN_Agent(input_shape=env.observation_space.shape, 
                         num_actions=env.action_space.n,
                         gamma=0.99,
                         epsilon=1,
                         epsilon_decay=0.999,
                         epsilon_min=0.05,
                         memory_type='',
                         buffer_size=2**15,
                         pretrained='')
num_episodes = 1000
beta_anneal_episodes = (200, 800)
replay_period = (1,100)
batch_size = 128

reward_list = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size)

fig_title = 'Dueling_DQN'
x_axis = np.arange(len(reward_list))+1
plt.figure(figsize=[3,3], dpi=300)
plt.title(fig_title, fontsize=9)
plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
plt.xlable('Episodes', fontsize=7)
plt.ylabel('Total Rewards', fontsize=7)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(linewidth=.1)
plt.savefig('figs/'+fig_title+'.png', bbox_inches='tight')
plt.close()


## Double Dueling DQN
env = gym.make('CartPole-v1')
agent = Double_Dueling_DQN_Agent(input_shape=env.observation_space.shape, 
                                num_actions=env.action_space.n,
                                gamma=0.99,
                                epsilon=1,
                                epsilon_decay=0.999,
                                epsilon_min=0.05,
                                memory_type='',
                                buffer_size=2**15,
                                pretrained='')
num_episodes = 1000
beta_anneal_episodes = (200, 800)
replay_period = (1,100)
batch_size = 128

reward_list = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size)

fig_title = 'Double_Dueling_DQN'
x_axis = np.arange(len(reward_list))+1
plt.figure(figsize=[3,3], dpi=300)
plt.title(fig_title, fontsize=9)
plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
plt.xlable('Episodes', fontsize=7)
plt.ylabel('Total Rewards', fontsize=7)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(linewidth=.1)
plt.savefig('figs/'+fig_title+'.png', bbox_inches='tight')
plt.close()

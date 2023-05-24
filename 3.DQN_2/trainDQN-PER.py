import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add
import matplotlib.pyplot as plt

from DQN_agents import DQN_Agent, Double_DQN_Agent, Dueling_DQN_Agent, Double_Dueling_DQN_Agent

def train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, repeats=1):
    
    agent_name = agent.__class__.__name__.split('_')[:-1]
    if not os.path.exists(agent_name):
        os.makedirs(agent_name)

    reward_array = np.zeros((repeats, num_episodes))

    for j in range(repeats):
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
                    agent.buffer.beta = agent.buffer.anneal_beta(episode, beta_anneal_episodes[0], beta_anneal_episodes[1], 0.4, 1)
                if episode > beta_anneal_episodes[1]:
                    agent.buffer.beta = 1

            reward_list.append(total_reward)
            print(f'Episode: {episode+1}, total reward: {total_reward}, buffer size: {agent.buffer.size()}')

        reward_array[j] = reward_list
        agent.save_model(os.path.join(agent_name + '/' + agent_name + '_' + str(j)))

        x_axis = np.arange(1, num_episodes+1)
        plt.figure(figsize=[3,3], dpi=300)
        plt.title(agent_name, fontsize=9)
        plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
        plt.xlabel('Episodes', fontsize=7)
        plt.ylabel('Total Rewards', fontsize=7)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.grid(linewidth=.1)
        plt.savefig('figs_'+agent.memory_type+'/'+agent_name+'_'+str(j)+'.png', bbox_inches='tight')
        plt.close()

    mean_reward_array = np.mean(reward_array, axis=0)

    x_axis = np.arange(1, num_episodes+1)
    plt.figure(figsize=[3,3], dpi=300)
    plt.title(agent_name, fontsize=9)
    plt.plot(x_axis, mean_reward_array, 'b-', linewidth=.5)
    plt.xlabel('Episodes', fontsize=7)
    plt.ylabel('Total Rewards', fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(linewidth=.1)
    plt.savefig('figs_'+agent.memory_type+'/'+agent_name+'average_'+str(repeats)+'.png', bbox_inches='tight')
    plt.close()

    return reward_array

env = gym.make('CartPole-v1')
buffer_size=2**14
num_episodes = 500
beta_anneal_episodes = (200, 500)
replay_period = (1,100)
batch_size = 128
repeats = 10

## DQN
agent = DQN_Agent(input_shape=env.observation_space.shape, 
                  num_actions=env.action_space.n,
                  gamma=0.99,
                  epsilon=1,
                  epsilon_decay=0.999,
                  epsilon_min=0.05,
                  memory_type='PER',
                  buffer_size=buffer_size,
                  pretrained='')

reward_array = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, repeats)


## Double DQN
agent = Double_DQN_Agent(input_shape=env.observation_space.shape, 
                         num_actions=env.action_space.n,
                         gamma=0.99,
                         epsilon=1,
                         epsilon_decay=0.999,
                         epsilon_min=0.05,
                         memory_type='PER',
                         buffer_size=buffer_size,
                         pretrained='')

reward_array = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, repeats)


## Dueling DQN
agent = Dueling_DQN_Agent(input_shape=env.observation_space.shape, 
                         num_actions=env.action_space.n,
                         gamma=0.99,
                         epsilon=1,
                         epsilon_decay=0.999,
                         epsilon_min=0.05,
                         memory_type='PER',
                         buffer_size=buffer_size,
                         pretrained='')

reward_array = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, repeats)


## Double Dueling DQN
agent = Double_Dueling_DQN_Agent(input_shape=env.observation_space.shape, 
                                num_actions=env.action_space.n,
                                gamma=0.99,
                                epsilon=1,
                                epsilon_decay=0.999,
                                epsilon_min=0.05,
                                memory_type='PER',
                                buffer_size=buffer_size,
                                pretrained='')

reward_array = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, repeats)

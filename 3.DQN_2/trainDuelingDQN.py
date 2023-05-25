import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from DQN_agents import DQN_Agent, DoubleDQN_Agent, DuelingDQN_Agent, DoubleDuelingDQN_Agent

def train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, index:int=0):
    
    agent_name = agent.__class__.__name__.split('_')[0]
    if not os.path.exists(agent_name):
        os.makedirs(agent_name)

    memory_type = agent.memory_type
    if memory_type:
        fig_path = agent_name+'/figs_PER/'
    else:
        fig_path = agent_name+'/figs/'

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    reward_list = []
    action_num = env.action_space.n
    one_hot_action = np.eye(action_num)
    
    for episode in range(num_episodes):

        counter = 0
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
            # agent.soft_update_target(TAU=0.005)

            state = next_state

        reward_list.append(total_reward)
        
        if agent.memory_type == 'PER':
            if episode >= beta_anneal_episodes[0]:
                agent.buffer.beta = agent.buffer.anneal_beta(episode, beta_anneal_episodes[0], beta_anneal_episodes[1], 0.4, 1)
            if episode > beta_anneal_episodes[1]:
                agent.buffer.beta = 1

        agent.epsilon = agent.linear_schedule_epsilon(episode=episode, max_episode = 300)
        
        print(f'Episode: {episode+1}, total reward: {total_reward}, buffer size: {agent.buffer.size()}, epsilon: {agent.epsilon}')

    if memory_type:
        agent.save_model(os.path.join(agent_name + '/' + agent_name + '_PER_' + str(index)))
    else:
        agent.save_model(os.path.join(agent_name + '/' + agent_name + '_' + str(index)))

    x_axis = np.arange(1, num_episodes+1)
    plt.figure(figsize=[3,3], dpi=300)
    plt.title(agent_name, fontsize=9)
    plt.plot(x_axis, reward_list, 'b-', linewidth=.5)
    plt.xlabel('Episodes', fontsize=7)
    plt.ylabel('Total Rewards', fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(linewidth=.1)
    plt.savefig(fig_path+agent_name+'_'+str(index)+'.png', bbox_inches='tight')
    plt.close()

    return reward_list

env = gym.make('CartPole-v1')
buffer_size=2**15
num_episodes = 600
beta_anneal_episodes = (200, 600)
replay_period = (1,200)
batch_size = 64
repeats = 3

reward_array = np.zeros((repeats, num_episodes))

## Dueling DQN
for i in range(repeats):
    agent = DuelingDQN_Agent(input_shape=env.observation_space.shape, 
                            num_actions=env.action_space.n,
                            gamma=0.99,
                            epsilon=1,
                            epsilon_min=0.1,
                            memory_type='',
                            buffer_size=buffer_size,
                            pretrained='')

    reward_list = train(env, agent, num_episodes, beta_anneal_episodes, replay_period, batch_size, i)
    reward_array[i] = reward_list

mean_reward_array = np.mean(reward_array, axis=0)

agent_name = agent.__class__.__name__.split('_')[0]
if agent.memory_type:
    fig_path = agent_name+'/figs_PER/'
else:
    fig_path = agent_name+'/figs/'
    
x_axis = np.arange(1, num_episodes+1)
plt.figure(figsize=[3,3], dpi=300)
plt.title(agent_name, fontsize=9)
plt.plot(x_axis, mean_reward_array, 'b-', linewidth=.5)
plt.xlabel('Episodes', fontsize=7)
plt.ylabel('Average Rewards', fontsize=7)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(linewidth=.1)
plt.savefig(fig_path+agent_name+'_average_'+str(repeats)+'.png', bbox_inches='tight')
plt.close()

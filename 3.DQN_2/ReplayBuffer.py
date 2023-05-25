import random
import numpy as np

from collections import deque
from SumTree import Node, create_tree, retrieve, leaf_update, propagate_changes

class ReplayBuffer():
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
        
    def replay_buffer_sampling(self, batch_size):
        experience_samples = random.sample(self.buffer, batch_size)
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = map(np.asarray, zip(*experience_samples))
        return state_arr, action_arr, reward_arr, next_state_arr, done_arr
    
    def size(self):
        return len(self.buffer)
    

class Prioritized_Experience_ReplayBuffer(object):
    def __init__(self, capacity:int=2**15):
        self.capacity = capacity                             # MUST be a power of 2 (binary tree)
        
        self.buffer = [[0, 0, 0, 0, bool]] * self.capacity   # (state, action, reward, next_state, done)
        self.root_node, self.leaf_nodes = create_tree([0 for i in range(self.capacity)])

        self.curr_write_idx = 0                          # the index of the buffer where the latest sample will be stored.
        self.available_samples = 0                       # the number of samples in the buffer. 
        self.beta = 0.4
        self.alpha = 0.6
        self.eps = 1e-5                                  # epsilon which makes priority nonzero
        self.max_priority = 1e-5
    
    def size(self):
        return self.available_samples

    def store(self, state, action, reward, next_state, done):
        self.buffer[self.curr_write_idx] = [state, action, reward, next_state, done]
        priority = self.max_priority                     # Set Maximal Priority for new experience
        self.update(self.curr_write_idx, priority)       # update the leaf node value (priority)
        
        self.curr_write_idx += 1
        if self.curr_write_idx >= self.capacity:
            self.curr_write_idx = 0
        if self.available_samples < self.capacity:
            self.available_samples += 1
            
    def update(self, idx: int, priority: float):
        if self.adjust_priority(priority) > self.max_priority:
            self.max_priority = self.adjust_priority(priority)
            
        leaf_update(self.leaf_nodes[idx], self.adjust_priority(priority))
        
    def adjust_priority(self, priority: float):
        return np.power(priority + self.eps, self.alpha)
    
    def replay_buffer_sampling(self, batch_size: int):
        
        sampled_idxs = [0]*batch_size                           # indices of samples in leaf nodes
        is_weights = [0]*batch_size                             # Importance Sampling Weights
        
        sub_length = self.root_node.value / batch_size         # Sub-interval Length
        sample_counter = 0
        
        while sample_counter < batch_size:
            temp_value = np.random.uniform(sub_length*sample_counter, sub_length*(sample_counter+1))
            sample_node = retrieve(temp_value, self.root_node)     # returns corresponding leaf node.
            
            # if sample_node.idx < self.available_samples:
            sampled_idxs[sample_counter] = sample_node.idx
            p = sample_node.value / self.root_node.value
            is_weights[sample_counter] = (self.available_samples) * p    # reciprocal of Importance Sampling weight
            sample_counter += 1
                
        # apply the beta factor and normalize weights by the maximum is_weight
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)
        
        sampled_replay = []
        for idx in sampled_idxs:
            sampled_replay.append(self.buffer[idx])
        
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = map(np.asarray, zip(*sampled_replay))
        return state_arr, action_arr, reward_arr, next_state_arr, done_arr, sampled_idxs, is_weights
    
    def anneal_beta(self, episode, start_episode, max_episode, start=0.4, end=1):
        return (start*(max_episode-episode) + end*(episode-start_episode)) / (max_episode - start_episode)
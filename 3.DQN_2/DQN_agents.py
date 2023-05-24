import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add

from ReplayBuffer import ReplayBuffer, Prioritized_Experience_ReplayBuffer

class DQN_Agent():
    def __init__(self, input_shape=(4,), num_actions=2, gamma=0.99, epsilon=0.25, epsilon_decay=0.99, epsilon_min=0.05, memory_type='PER', buffer_size=2**15, pretrained=''):
        
        self.num_actions = num_actions
        self.agent = self.nn_model(input_size=input_shape, action_dim=num_actions)
        self.target_agent = self.nn_model(input_size=input_shape, action_dim=num_actions)
        self.update_target()
        self.optimizer = tf.keras.optimizers.Adam()

        self.memory_type = memory_type
        if memory_type == 'PER':
            self.buffer = Prioritized_Experience_ReplayBuffer(capacity=buffer_size)
        else:
            self.buffer = ReplayBuffer(capacity=buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if pretrained:
            self.continue_training(pretrained)

    def nn_model(self, input_size, action_dim):
    
        input_layer = Input(shape=input_size)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)
        output_layer = Dense(action_dim, activation='linear')(x)

        model = Model(input_layer, outputs = output_layer)

        return model
    
    def update_target(self):
        self.target_agent.set_weights(self.agent.get_weights())

    def replay_experience(self, batch_size):
        if self.memory_type == 'PER':
            with tf.GradientTape() as tape:
                state_arr, action_arr, reward_arr, next_state_arr, done_arr, sampled_idxs, is_weights = self.buffer.replay_buffer_sampling(batch_size)
                predicts = tf.reduce_sum(self.agent(state_arr, training=True)*action_arr, axis=1)
                next_q_values = np.max(self.target_agent(next_state_arr, training=False), axis=1)
                targets = reward_arr + self.gamma*next_q_values*(1-done_arr)
                td = targets - predicts
                loss = tf.reduce_mean(is_weights * td**2)
            grads = tape.gradient(loss, self.agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))
            
            return abs(td), sampled_idxs
        
        else:
            with tf.GradientTape() as tape:
                state_arr, action_arr, reward_arr, next_state_arr, done_arr = self.buffer.replay_buffer_sampling(batch_size)
                predicts = tf.reduce_sum(self.agent(state_arr, training=True)*action_arr, axis=1)
                next_q_values = np.max(self.target_agent(next_state_arr, training=False), axis=1)
                targets = reward_arr + self.gamma*next_q_values*(1-done_arr)
                td = targets - predicts
                loss = tf.reduce_mean(td**2)
                
            grads = tape.gradient(loss, self.agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

            return 

    def get_action(self, observation):
        observation = observation.reshape(1,-1)
        action_logits = self.agent.predict_on_batch(tf.convert_to_tensor(observation))

        should_explore = np.random.rand()
        if should_explore < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(action_logits, axis=1)[0]
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return action

    def save_model(self, mdir):
        self.agent.save_weights(mdir)

    def continue_training(self, mdir):
        self.agent.load_weights(mdir)
    
class DoubleDQN_Agent(DQN_Agent):
    def replay_experience(self, batch_size):
        if self.memory_type == 'PER':
            with tf.GradientTape() as tape:
                state_arr, action_arr, reward_arr, next_state_arr, done_arr, sampled_idxs, is_weights = self.buffer.replay_buffer_sampling(batch_size)
                predicts = tf.reduce_sum(self.agent(state_arr, training=True)*action_arr, axis=1)

                next_q_targets = self.target_agent(next_state_arr, training=False)
                next_q_values  = next_q_targets.numpy()[range(batch_size), np.argmax(self.agent(next_state_arr), axis=1)]

                targets = reward_arr + self.gamma*next_q_values*(1-done_arr)
                td = targets - predicts
                loss = tf.reduce_mean(is_weights * td**2)
            grads = tape.gradient(loss, self.agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))
            
            return abs(td), sampled_idxs
        
        else:
            with tf.GradientTape() as tape:
                state_arr, action_arr, reward_arr, next_state_arr, done_arr = self.buffer.replay_buffer_sampling(batch_size)
                predicts = tf.reduce_sum(self.agent(state_arr, training=True)*action_arr, axis=1)
                
                next_q_targets = self.target_agent(next_state_arr, training=False)
                next_q_values = next_q_targets.numpy()[range(batch_size), np.argmax(self.agent(next_state_arr), axis=1)]

                targets = reward_arr + self.gamma*next_q_values*(1-done_arr)
                td = targets - predicts
                loss = tf.reduce_mean(td**2)
            grads = tape.gradient(loss, self.agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

            return

    def get_action(self, observation):
        
        obs_tf = tf.convert_to_tensor(observation.reshape(1,-1))
        action_logits1 = self.agent.predict_on_batch(obs_tf)
        action_logits2 = self.target_agent.predict_on_batch(obs_tf)

        action_logits = action_logits1 + action_logits2

        should_explore = np.random.rand()
        if should_explore < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(action_logits, axis=1)[0]
        self.epsilon *= self.epsilon_decay

        return action
    
class DuelingDQN_Agent(DQN_Agent):
    def nn_model(self, input_size, action_dim):
        
        input_layer = Input(shape=input_size)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)

        v_out = Dense(1, activation='linear')(x)
        adv_out = Dense(action_dim, activation='linear')(x)
        adv_mean = -tf.reduce_mean(adv_out, axis=1)

        output_layer = Add()([v_out, adv_out, adv_mean])
        model = Model(input_layer, outputs = output_layer)

        return model
    
class DoubleDuelingDQN_Agent(DoubleDQN_Agent):
    def nn_model(self, input_size, action_dim):
        
        input_layer = Input(shape=input_size)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)

        v_out = Dense(1, activation='linear')(x)
        adv_out = Dense(action_dim, activation='linear')(x)
        adv_mean = -tf.reduce_mean(adv_out, axis=1)

        output_layer = Add()([v_out, adv_out, adv_mean])
        model = Model(input_layer, outputs = output_layer)

        return model
    
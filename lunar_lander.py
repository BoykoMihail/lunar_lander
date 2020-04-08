import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


class Deep_Q_Learning_Model:

    def __init__(self, output_dim, input_dim):

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.batch_size = 64
        self.lr = 0.001
        self.epsilon = 1.0
        self.gamma = 0.99
        self.memory = deque(maxlen=1000000)
        
        self.model = Sequential()
        self.model.add(Dense(150, input_dim=self.input_dim, activation=relu))
        self.model.add(Dense(70, activation=relu))
        self.model.add(Dense(self.output_dim, activation=linear))
        self.model.compile(loss='mse', optimizer=adam(lr=self.lr))
        
    def getMiniBatchItems(self, minibatch):
    
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        return states, actions, rewards, next_states, dones
        

    def recount(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = self.getMiniBatchItems(minibatch)

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > 0.01:
            self.epsilon *= 0.99
    
    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dim)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
            
    def tarain(self, lunar, episodes):
        for e in range(episodes):
            state = lunar.reset()
            state = np.reshape(state, (1, 8))
            score = 0
            max_steps = 1000
            for i in range(max_steps):
                action = self.getAction(state)
                lunar.render()
                next_state, reward, done, _ = lunar.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, 8))
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.recount()
                if done:
                    print("episode: {} from {}, score: {}".format(e, episodes, score))
                    break


if __name__ == '__main__':

    episodes = 1000
    agent = Deep_Q_Learning_Model(env.action_space.n, env.observation_space.shape[0])
    agent.tarain(env, episodes)
    print("Train is over!")
    

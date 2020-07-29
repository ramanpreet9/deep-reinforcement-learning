from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

#in v2 online on the notebook
# sarsa[0]   -->  9.256
# Q-learning -->  9.223
# E-Sarsa    -->  9.118


#in v3:
# sarsa[0]   -->  8.793
# Q-learning -->  8.554
# E-Sarsa    -->  8.874

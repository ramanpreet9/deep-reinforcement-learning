import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

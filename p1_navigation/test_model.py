from unityagents import UnityEnvironment
import numpy as np
from google.protobuf import descriptor as _descriptor
from dqns import dqn
import matplotlib.pyplot as plt
from dqn_agent import Agent
import torch

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", worker_id=3)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0] 

action_size = brain.vector_action_space_size
state_size = len(state)

#agent = Agent(state_size=state_size, action_size=action_size, seed=0)
#agent.load_state_dict(torch.load('checkpoint.pth'))
agent = torch.load('checkpoint.pth')
#agent.eval()

score = 0                                          # initialize the score
while True:
    action = agent.act(state, eps)
    action = action.astype(int)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))

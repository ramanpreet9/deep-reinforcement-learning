from unityagents import UnityEnvironment
import numpy as np
from google.protobuf import descriptor as _descriptor
from dqns import dqn
import matplotlib.pyplot as plt
from dqn_agent import Agent

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", worker_id=2)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)





agent = Agent(state_size=state_size, action_size=action_size, seed=0)
scores = dqn(env, agent, brain_name)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



env.close()
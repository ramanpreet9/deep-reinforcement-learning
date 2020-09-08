from unityagents import UnityEnvironment
import numpy as np
#from ddpg_agent import Agent
from Agent_SV import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt

def ddpg(env, brain_name, agent, n_episodes=1000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    scores = []
    last_max_mean_score = -np.inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        #agent.reset()
        states = env_info.vector_observations 
        scores_local = np.zeros(num_agents)

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            #print('rewards =', rewards)
            dones = env_info.local_done                        # see if episode finished
            scores_local += np.array(env_info.rewards) 
            #scores_local = [x + y for x, y in zip(scores_local, env_info.rewards)]
            #for i in range(num_agents):
            
            if any(dones):
                print('breaking at  t= ', t, 'dones = ', dones)
                break 
            #agent.step(t, states, actions, rewards, next_states, dones)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states

        scores_deque.append(np.mean(scores_local))
        scores.append(np.mean(scores_local))
        
        
        print('\rEpisode {}\t, step {}\t,local score {:.2f}\t, Average Score: {:.2f}'.format(i_episode, t,np.mean(scores_local), np.mean(scores_deque)), end="")
        mean_score = np.mean(scores_deque)
        if mean_score > last_max_mean_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            last_max_mean_score = mean_score
            if last_max_mean_score > 30:
                return scores
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores




#env = UnityEnvironment(file_name='Reacher_Windows_x86_64_single/Reacher.exe')
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


#env_info = env.reset(train_mode=True)[brain_name] 
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)#, random_seed=66)
#agent.reset()
scores = ddpg(env, brain_name, agent)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

tt = input('random')

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()


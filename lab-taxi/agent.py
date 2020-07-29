import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.argmax(self.Q[state])
        #return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        '''
        # SARSA[0]
        if done:
            self.Q[state][action] += (reward - self.Q[state][action])
        else:
            next_action = self.select_action(state)
            self.Q[state][action] += (reward + self.Q[next_state][next_action]-self.Q[state][action])
        '''

        '''
        # Q -learning
        if done:
            Qs_max = 0
        else:
            Qs_max = np.max(self.Q[next_state])
        self.Q[state][action] += (reward + Qs_max - self.Q[state][action])
        '''

        # Expected SARSA
        if done:
            Qs_mean = 0
        else:
            Qs_mean = np.mean(self.Q[next_state])
        self.Q[state][action] += (reward + Qs_mean - self.Q[state][action])
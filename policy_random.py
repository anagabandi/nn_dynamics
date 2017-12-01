import numpy as np

class Policy_Random(object):

    def __init__(self, env):

        #vars
        self.env = env
        self.low_val = self.env.action_space.low
        self.high_val = self.env.action_space.high
        self.shape = self.env.action_space.shape
        print("Created a random policy, where actions are selected between ", self.low_val, ", and ", self.high_val)
        
    def get_action(self, observation):
        return np.random.uniform(self.low_val, self.high_val, self.shape), 0
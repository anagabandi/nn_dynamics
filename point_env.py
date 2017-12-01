from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

class PointEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(4,)) #state space = [x, y, vx, vy]

    @property
    def action_space(self):
        return Box(low=-5, high=5, shape=(2,)) #controls are the forces applied to pointmass

    def reset(self, init_state=None):
        if(init_state==None):
            np.random.seed()
            self._state=np.zeros((4,))
            self._state[0]= np.random.uniform(-10, 10)
            self._state[1]= np.random.uniform(-10, 10)
        else:
            self._state = init_state

        observation = np.copy(self._state)
        return observation

    def step(self, action):
        #next state = what happens after taking "action"
        temp_state=np.copy(self._state)
        dt=0.1
        temp_state[0] = self._state[0] + self._state[2]*dt + 0.5*action[0]*dt*dt
        temp_state[1] = self._state[1] + self._state[3]*dt + 0.5*action[1]*dt*dt
        temp_state[2] = self._state[2] + action[0]*dt
        temp_state[3] = self._state[3] + action[1]*dt
        self._state = np.copy(temp_state)

        #make the reward what you care about
        x, y, vx, vy = self._state
        reward = vx - np.sqrt(abs(y-0)) #we care about moving in the forward x direction... and keeping our y value close to 0... (aka "going straight")
        done = 0#x>500 #when do you consider this to be "done" (rollout stops... "terminal")
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        return self._state
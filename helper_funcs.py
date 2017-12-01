import copy
import time
import tensorflow as tf
import numpy as np 

#import rllab envs
from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from point_env import PointEnv
from rllab.envs.mujoco.ant_env import AntEnv

#import gym envs
import gym
from gym import wrappers
from gym.envs.mujoco.reacher import ReacherEnv
from rllab.envs.gym_env import GymEnv


def add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[0]):
        if(std_of_noise[j]>0):
            data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data

def perform_rollouts(policy, num_rollouts, steps_per_rollout, visualize_rollouts, CollectSamples, 
                    env, which_agent, dt_steps, dt_from_xml, follow_trajectories):
    #collect training data by performing rollouts
    print("Beginning to do ", num_rollouts, " rollouts.")
    c = CollectSamples(env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories)
    states, controls, starting_states, rewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.")
    return states, controls, starting_states, rewards_list


def create_env(which_agent):

    # setup environment
    if(which_agent==0):
        env = normalize(PointEnv())
    elif(which_agent==1):
        env = normalize(AntEnv())
    elif(which_agent==2):
        env = normalize(SwimmerEnv()) #dt 0.001 and frameskip=150
    elif(which_agent==3):
        env = ReacherEnv() 
    elif(which_agent==4):
        env = normalize(HalfCheetahEnv())
    elif(which_agent==5):
        env = RoachEnv() #this is a personal vrep env
    elif(which_agent==6):
        env=normalize(HopperEnv())
    elif(which_agent==7):
        env=normalize(Walker2DEnv())

    #get dt value from env
    if(which_agent==5):
        dt_from_xml = env.VREP_DT
    else:
        dt_from_xml = env.model.opt.timestep
    print("\n\n the dt is: ", dt_from_xml, "\n\n")

    #set vars
    tf.set_random_seed(2)
    gym.logger.setLevel(gym.logging.WARNING)
    dimO = env.observation_space.shape
    dimA = env.action_space.shape
    print ('--------------------------------- \nState space dimension: ', dimO)
    print ('Action space dimension: ', dimA, "\n -----------------------------------")

    return env, dt_from_xml


def visualize_rendering(starting_state, list_of_actions, env_inp, dt_steps, dt_from_xml, which_agent):
    env=copy.deepcopy(env_inp)

    if(which_agent==5):
        env.reset()
    else:
        env.reset(starting_state)

    for action in list_of_actions:

        if(action.shape[0]==1):
            env.step(action[0], collectingInitialData=False)
        else:
            env.step(action, collectingInitialData=False)

        if(which_agent==5):
            junk=1
        else:
            env.render()
            time.sleep(dt_steps*dt_from_xml)

    print("Done rendering.")
    return
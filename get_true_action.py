import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import copy
from six.moves import cPickle
from rllab.misc import tensor_utils
from rllab.envs.normalized_env import normalize
from feedforward_network import feedforward_network
import os
from data_manipulation import from_observation_to_usablestate
from dynamics_model import Dyn_Model
from data_manipulation import get_indices
from mpc_controller import MPCController
from trajectories import make_trajectory

class GetTrueAction:

    def make_model(self, sess, env_inp, rundir, tf_datatype, num_fc_layers, depth_fc_layers, which_agent, 
                    lr, batchsize, N, horizon, steps_per_episode, dt_steps, print_minimal):
        
        #vars
        self.sess = sess
        self.env = copy.deepcopy(env_inp)
        self.N = N 
        self.horizon = horizon
        self.which_agent = which_agent
        self.steps_per_episode = steps_per_episode
        self.dt_steps = dt_steps
        self.print_minimal = print_minimal

        #get sizes
        dataX= np.load(rundir + '/training_data/dataX.npy')
        dataY= np.load(rundir + '/training_data/dataY.npy')
        dataZ= np.load(rundir + '/training_data/dataZ.npy')
        inputs = np.concatenate((dataX, dataY), axis=1)
        assert inputs.shape[0] == dataZ.shape[0]
        inputSize = inputs.shape[1]
        outputSize = dataZ.shape[1]

        #calculate the means and stds
        self.mean_x = np.mean(dataX, axis = 0)
        dataX = dataX - self.mean_x
        self.std_x = np.std(dataX, axis = 0)
        dataX = np.nan_to_num(dataX/self.std_x)
        self.mean_y = np.mean(dataY, axis = 0) 
        dataY = dataY - self.mean_y
        self.std_y = np.std(dataY, axis = 0)
        dataY = np.nan_to_num(dataY/self.std_y)
        self.mean_z = np.mean(dataZ, axis = 0) 
        dataZ = dataZ - self.mean_z
        self.std_z = np.std(dataZ, axis = 0)
        dataZ = np.nan_to_num(dataZ/self.std_z)

        #get x and y index
        x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(which_agent)

        #make dyn model and randomly initialize weights
        self.dyn_model = Dyn_Model(inputSize, outputSize, self.sess, lr, batchsize, which_agent, x_index, y_index, num_fc_layers, 
                                    depth_fc_layers, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z, 
                                    tf_datatype, self.print_minimal)
        self.sess.run(tf.global_variables_initializer())

        #load in weights from desired trained dynamics model
        pathname = rundir + '/models/finalModel.ckpt'
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.sess, pathname)
        print("\n\nRestored dynamics model with variables from ", pathname,"\n\n")

        #make controller, to use for querying optimal action
        self.mpc_controller = MPCController(self.env, self.dyn_model, self.horizon, self.which_agent, self.steps_per_episode, 
                                            self.dt_steps, self.N, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, 
                                            self.std_z, 'nc', self.print_minimal, x_index, y_index, z_index, yaw_index, joint1_index, 
                                            joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index)
        self.mpc_controller.desired_states = make_trajectory('straight', np.zeros((100,)), x_index, y_index, which_agent) #junk, just a placeholder

        #select task or reward func
        self.reward_func = self.mpc_controller.reward_functions.get_reward_func(False, 0, 0, 0, 0)

    def get_action(self, curr_obs):

        curr_nn_state= from_observation_to_usablestate(curr_obs, self.which_agent, True)
        best_action, _, _, _ = self.mpc_controller.get_action(curr_nn_state, 0, self.reward_func)

        return best_action
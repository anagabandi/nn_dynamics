import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import copy
from six.moves import cPickle
from rllab.misc import tensor_utils
from data_manipulation import from_observation_to_usablestate
from reward_functions import RewardFunctions

class MPCController:

    def __init__(self, env_inp, dyn_model, horizon, which_agent, steps_per_episode, dt_steps, num_control_samples, 
                mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index, z_index, yaw_index, 
                joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index):

        #init vars
        self.env=copy.deepcopy(env_inp)
        self.N = num_control_samples
        self.which_agent = which_agent
        self.horizon = horizon
        self.dyn_model = dyn_model
        self.steps_per_episode = steps_per_episode 
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index 
        self.yaw_index = yaw_index
        self.joint1_index = joint1_index
        self.joint2_index = joint2_index
        self.frontleg_index = frontleg_index 
        self.frontshin_index = frontshin_index 
        self.frontfoot_index = frontfoot_index 
        self.xvel_index = xvel_index 
        self.orientation_index = orientation_index
        self.actions_ag = actions_ag
        self.print_minimal = print_minimal
        self.reward_functions = RewardFunctions(self.which_agent, self.x_index, self.y_index, self.z_index, self.yaw_index, 
                                                self.joint1_index, self.joint2_index, self.frontleg_index, self.frontshin_index, 
                                                self.frontfoot_index, self.xvel_index, self.orientation_index)       

    def perform_rollout(self, starting_fullenvstate, starting_observation, starting_observation_NNinput, desired_states, follow_trajectories, 
                        horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, noise_actions, noise_amount):
        
        #lists for saving info
        traj_taken=[] #list of states that go into NN
        actions_taken=[]
        observations = [] #list of observations (direct output of the env)
        rewards = []
        agent_infos = []
        env_infos = []

        #init vars
        stop_taking_steps = False
        total_reward_for_episode = 0
        step=0
        curr_line_segment = 0
        self.horiz_penalty_factor = horiz_penalty_factor
        self.forward_encouragement_factor = forward_encouragement_factor
        self.heading_penalty_factor = heading_penalty_factor

        #extend the list of desired states so you don't run out
        temp = np.tile(np.expand_dims(desired_states[-1], axis=0), (10,1))
        self.desired_states = np.concatenate((desired_states, temp))

        #reset env to the given full env state
        if(self.which_agent==5):
            self.env.reset()
        else:
            self.env.reset(starting_fullenvstate)

        #current observation
        obs = np.copy(starting_observation)
        #current observation in the right format for NN
        curr_state = np.copy(starting_observation_NNinput)
        traj_taken.append(curr_state)

        #select task or reward func
        reward_func = self.reward_functions.get_reward_func(follow_trajectories, self.desired_states, horiz_penalty_factor, 
                                                            forward_encouragement_factor, heading_penalty_factor)

        #take steps according to the chosen task/reward function
        while(stop_taking_steps==False):

            #get optimal action
            best_action, best_sim_number, best_sequence, moved_to_next = self.get_action(curr_state, curr_line_segment, reward_func)

            #advance which line segment we are on
            if(follow_trajectories):
                if(moved_to_next[best_sim_number]==1):
                    curr_line_segment+=1
                    print("MOVED ON TO LINE SEGMENT ", curr_line_segment)

            #noise the action
            action_to_take= np.copy(best_action)

            #whether to execute noisy or clean actions
            if(self.actions_ag=='nn'):
                noise_actions=True
            if(self.actions_ag=='nc'):
                noise_actions=True
            if(self.actions_ag=='cc'):
                noise_actions=False

            clean_action = np.copy(action_to_take)
            if(noise_actions):
                noise = noise_amount * npr.normal(size=action_to_take.shape)#
                action_to_take = action_to_take + noise
                action_to_take=np.clip(action_to_take, -1,1)

            #execute the action
            next_state, rew, done, env_info = self.env.step(action_to_take, collectingInitialData=False)

            #check if done
            if(done):
                stop_taking_steps=True
            else:
                #save things
                observations.append(obs)
                rewards.append(rew)
                env_infos.append(env_info)
                total_reward_for_episode += rew

                #whether to save clean or noisy actions
                if(self.actions_ag=='nn'):
                    actions_taken.append(np.array([action_to_take]))
                if(self.actions_ag=='nc'):
                    actions_taken.append(np.array([clean_action]))
                if(self.actions_ag=='cc'):
                    actions_taken.append(np.array([clean_action]))

                #this is the observation returned by taking a step in the env
                obs=np.copy(next_state)

                #get the next state (usable by NN)
                just_one=True
                next_state = from_observation_to_usablestate(next_state, self.which_agent, just_one)
                curr_state=np.copy(next_state)
                traj_taken.append(curr_state)

                #bookkeeping
                if(not(self.print_minimal)):
                    if(step%100==0):
                        print("done step ", step, ", rew: ", total_reward_for_episode)
                step+=1

                #when to stop
                if(follow_trajectories):
                    if((step>=self.steps_per_episode) or (curr_line_segment>5)):
                        stop_taking_steps = True
                else:
                    if(step>=self.steps_per_episode):
                        stop_taking_steps = True

        if(not(self.print_minimal)):
            print("DONE TAKING ", step, " STEPS.")
            print("Reward: ", total_reward_for_episode)

        mydict = dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions_taken),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=agent_infos,
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos))

        return traj_taken, actions_taken, total_reward_for_episode, mydict

    def get_action(self, curr_nn_state, curr_line_segment, reward_func):
        #randomly sample N candidate action sequences
        all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))

        #forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(all_samples), many_in_parallel, self.env, self.which_agent)
        resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]

        #init vars to evaluate the trajectories
        scores=np.zeros((self.N,))
        done_forever=np.zeros((self.N,))
        move_to_next=np.zeros((self.N,))
        curr_seg = np.tile(curr_line_segment,(self.N,))
        curr_seg = curr_seg.astype(int)
        prev_forward = np.zeros((self.N,))
        moved_to_next = np.zeros((self.N,))
        prev_pt = resulting_states[0]

        #accumulate reward over each timestep
        for pt_number in range(resulting_states.shape[0]):

            #array of "the point"... for each sim
            pt = resulting_states[pt_number] # N x state

            #how far is the point from the desired trajectory
            #how far along the desired traj have you moved since the last point
            min_perp_dist, curr_forward, curr_seg, moved_to_next = self.calculate_geometric_trajfollow_quantities(pt, curr_seg, moved_to_next)

            #update reward score
            scores, done_forever = reward_func(pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, curr_seg, 
                                                moved_to_next, done_forever, all_samples, pt_number)

            #update vars
            prev_forward = np.copy(curr_forward)
            prev_pt = np.copy(pt)

        #pick best action sequence
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores) 
        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])

        

        return best_action, best_sim_number, best_sequence, moved_to_next

    def calculate_geometric_trajfollow_quantities(self, pt, curr_seg, moved_to_next):

        #arrays of line segment points... for each sim
        curr_start = self.desired_states[curr_seg]
        curr_end = self.desired_states[curr_seg+1]
        next_start = self.desired_states[curr_seg+1]
        next_end = self.desired_states[curr_seg+2]

        #initialize
        min_perp_dist = np.ones((self.N, ))*5000

        ####################################### closest distance from point to current line segment

        #vars
        a = pt[:,self.x_index]- curr_start[:,0]
        b = pt[:,self.y_index]- curr_start[:,1]
        c = curr_end[:,0]- curr_start[:,0]
        d = curr_end[:,1]- curr_start[:,1]

        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

        #point on line segment that's closest to the pt
        closest_pt_x = np.copy(which_line_section)
        closest_pt_y = np.copy(which_line_section)
        closest_pt_x[which_line_section<0] = curr_start[:,0][which_line_section<0]
        closest_pt_y[which_line_section<0] = curr_start[:,1][which_line_section<0]
        closest_pt_x[which_line_section>1] = curr_end[:,0][which_line_section>1]
        closest_pt_y[which_line_section>1] = curr_end[:,1][which_line_section>1]
        closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,0] + 
                            np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
        closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,1] + 
                            np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        min_perp_dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
                                (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

        ####################################### "forward-ness" of the pt... for each sim
        curr_forward = which_line_section

        ###################################### closest distance from point to next line segment

        #vars
        a = pt[:,self.x_index]- next_start[:,0]
        b = pt[:,self.y_index]- next_start[:,1]
        c = next_end[:,0]- next_start[:,0]
        d = next_end[:,1]- next_start[:,1]

        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), 
                                        (np.multiply(c,c) + np.multiply(d,d)))

        #point on line segment that's closest to the pt
        closest_pt_x = np.copy(which_line_section)
        closest_pt_y = np.copy(which_line_section)
        closest_pt_x[which_line_section<0] = next_start[:,0][which_line_section<0]
        closest_pt_y[which_line_section<0] = next_start[:,1][which_line_section<0]
        closest_pt_x[which_line_section>1] = next_end[:,0][which_line_section>1]
        closest_pt_y[which_line_section>1] = next_end[:,1][which_line_section>1]
        closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,0] + 
                            np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
        closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,1] + 
                            np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
                        (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

        ############################################ 

        #pick which line segment it's closest to, and update vars accordingly
        curr_seg[dist<=min_perp_dist] += 1
        moved_to_next[dist<=min_perp_dist] = 1
        curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]
        min_perp_dist = np.min([min_perp_dist, dist], axis=0)

        return min_perp_dist, curr_forward, curr_seg, moved_to_next
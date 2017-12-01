import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import copy
import multiprocessing

class CollectSamples(object):

    def __init__(self, env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories):
        self.main_env = copy.deepcopy(env)
        self.policy = policy
        self.visualize_at_all = visualize_rollouts
        self.which_agent = which_agent
        self.list_observations=[]
        self.list_actions=[]
        self.list_starting_states=[]

        self.stateDim = self.main_env.observation_space.shape[0]
        self.actionDim = self.main_env.action_space.shape[0]

        self.dt_steps = dt_steps
        self.dt_from_xml = dt_from_xml
        self.follow_trajectories = follow_trajectories

    def collect_samples(self, num_rollouts, steps_per_rollout):
        
        #vars
        all_processes=[]
        visualization_frequency = num_rollouts/10
        num_workers=multiprocessing.cpu_count() #detect number of cores
        pool = multiprocessing.Pool(8)

        #multiprocessing for running rollouts (utilize multiple cores)
        for rollout_number in range(num_rollouts):
            result = pool.apply_async(self.do_rollout, 
                                    args=(steps_per_rollout, rollout_number, visualization_frequency), 
                                    callback=self.mycallback)

        pool.close() #not going to add anything else to the pool
        pool.join() #wait for the processes to terminate

        #return lists of length = num rollouts
        #each entry contains one rollout
        #each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
        return self.list_observations, self.list_actions, self.list_starting_states, []

    def mycallback(self, x): #x is shape [numSteps, state + action]
        self.list_observations.append(x[:,0:self.stateDim])
        self.list_actions.append(x[:,self.stateDim:(self.stateDim+self.actionDim)])
        self.list_starting_states.append(x[0,(self.stateDim+self.actionDim):])

    def do_rollout(self, steps_per_rollout, rollout_number, visualization_frequency):
        #init vars
        #print("START ", rollout_number)
        observations = []
        actions = []
        visualize = False

        env = copy.deepcopy(self.main_env)

        #reset env
        if(self.which_agent==2):
            if(self.follow_trajectories):
                observation, starting_state = env.reset(returnStartState=True, isSwimmer=True, need_diff_headings=True)
            else:
                observation, starting_state = env.reset(returnStartState=True, isSwimmer=True)
        else:
            observation, starting_state = env.reset(returnStartState=True)

        #visualize only sometimes
        if((rollout_number%visualization_frequency)==0):
            if(self.visualize_at_all):
                all_states=[]
                print ("---- visualizing a rollout ----")
                visualize=True

        for step_num in range(steps_per_rollout):

            #decide what action to take
            action, _ = self.policy.get_action(observation)

            #keep tracks of observations + actions
            observations.append(observation)
            actions.append(action)

            #perform the action
            next_observation, reward, terminal, _ = env.step(action, collectingInitialData=True)

            #update the observation
            observation = np.copy(next_observation)
            
            if terminal:
                #print("Had to stop rollout because terminal state was reached.")
                break

            if(visualize):
                if(self.which_agent==0):
                    curr_state = env.render()
                    all_states.append(np.expand_dims(curr_state, axis=0))
                else:
                    env.render()
                    time.sleep(self.dt_steps*self.dt_from_xml)

        if(visualize and (self.which_agent==0)):
            all_states= np.concatenate(all_states, axis=0)
            plt.plot(all_states[:,0], all_states[:,1], 'r')
            plt.show()

        if((rollout_number%visualization_frequency)==0):
            print("Completed rollout # ", rollout_number)

        array_starting_state = np.tile(starting_state, (np.array(actions).shape[0],1))
        return np.concatenate((np.array(observations), np.array(actions), array_starting_state), axis=1)
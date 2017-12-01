import numpy as np 
import matplotlib.pyplot as plt
import math
npr = np.random
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import tensorflow as tf 
from six.moves import cPickle
from collect_samples import CollectSamples
from get_true_action import GetTrueAction
import os
import copy
from helper_funcs import create_env
from helper_funcs import perform_rollouts
from helper_funcs import add_noise
from feedforward_network import feedforward_network
from helper_funcs import visualize_rendering
import argparse

#TRPO things
from rllab.envs.normalized_env import normalize
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from rllab.misc.instrument import run_experiment_lite

def nn_policy(inputState, junk1, outputSize, junk2, junk3, junk4):
	#init vars
	x = inputState
	initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float64)
	fc = tf.contrib.layers.fully_connected
	weights_reg = tf.contrib.layers.l2_regularizer(scale=0.001)
	#hidden layer 1
	fc1 = fc(x, num_outputs= 64, activation_fn=None, trainable=True, reuse=False, weights_initializer=initializer, 
			biases_initializer=initializer, weights_regularizer=weights_reg)
	h1 = tf.tanh(fc1)
	#hidden layer 2
	fc2 = fc(h1, num_outputs= 64, activation_fn=None, trainable=True, reuse=False, weights_initializer=initializer, 
			biases_initializer=initializer, weights_regularizer=weights_reg)
	h2 = tf.tanh(fc2)
	# output layer
	output = fc(h2, num_outputs=outputSize, activation_fn=None, trainable=True, reuse=False, 
				weights_initializer=initializer, biases_initializer=initializer)
	return output

def run_task(v):

	which_agent=v["which_agent"]
	env,_ = create_env(which_agent)
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	optimizer_params = dict(base_eps=1e-5)

	#how many iters
	num_trpo_iters = 2500
	if(which_agent==1):
		num_trpo_iters = 2500
	if(which_agent==2):
		steps_per_rollout=333
		num_trpo_iters = 200
	if(which_agent==4):
		num_trpo_iters= 2000
	if(which_agent==6):
		num_trpo_iters= 2000

	#recreate the policy
	policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(v["depth_fc_layers"], v["depth_fc_layers"]), init_std=v["std_on_mlp_policy"])
	all_params = np.concatenate((v["policy_values"], policy._l_log_std.get_params()[0].get_value()))
	policy.set_param_values(all_params)
	

	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		batch_size=v["trpo_batchsize"],
		max_path_length=v["steps_per_rollout"],
		n_itr=num_trpo_iters,
		discount=0.995,
		optimizer=v["ConjugateGradientOptimizer"](hvp_approach=v["FiniteDifferenceHvp"](**optimizer_params)),
		step_size=0.05,
		plot_true=True)

	#train the policy
	algo.train()

##########################################
##########################################

#ARGUMENTS TO SPECIFY
parser = argparse.ArgumentParser()
parser.add_argument('--save_trpo_run_num', type=int, default='1')
parser.add_argument('--run_num', type=int, default=1)
parser.add_argument('--which_agent', type=int, default=1)
parser.add_argument('--std_on_mlp_policy', type=float, default=0.5)
parser.add_argument('--num_workers_trpo', type=int, default=2)
parser.add_argument('--might_render', action="store_true", dest='might_render', default=False)
parser.add_argument('--visualize_mlp_policy', action="store_true", dest='visualize_mlp_policy', default=False)
parser.add_argument('--visualize_on_policy_rollouts', action="store_true", dest='visualize_on_policy_rollouts', default=False)
parser.add_argument('--print_minimal', action="store_true", dest='print_minimal', default=False)
parser.add_argument('--use_existing_pretrained_policy', action="store_true", dest='use_existing_pretrained_policy', default=False)
args = parser.parse_args()

##########################################
##########################################

#save args
save_trpo_run_num= args.save_trpo_run_num
run_num = args.run_num
which_agent = args.which_agent
visualize_mlp_policy = args.visualize_mlp_policy
visualize_on_policy_rollouts = args.visualize_on_policy_rollouts
print_minimal = args.print_minimal
std_on_mlp_policy = args.std_on_mlp_policy

#swimmer
trpo_batchsize = 50000
if(which_agent==2):
	#training vars for new policy
	batchsize = 512
	nEpoch = 70
	learning_rate = 0.001
	#aggregation for training of new policy
	num_agg_iters = 3
	num_rollouts_to_agg= 5
	num_rollouts_testperformance = 2
	start_using_noised_actions = 0
	#other
	do_trpo = True
#cheetah
if(which_agent==4):
	#training vars for new policy
	batchsize = 512
	nEpoch = 300
	learning_rate = 0.001
	#aggregation for training of new policy
	num_agg_iters = 3
	num_rollouts_to_agg= 2
	num_rollouts_testperformance = 2
	start_using_noised_actions = 10
	#other
	do_trpo = True
#hopper
if(which_agent==6):
	#training vars for new policy
	batchsize = 512
	nEpoch = 200 #70
	learning_rate = 0.001
	#aggregation for training of new policy
	num_agg_iters = 5 #10
	num_rollouts_to_agg= 5 ###10
	num_rollouts_testperformance = 3
	start_using_noised_actions = 50
	#other
	do_trpo = True
	trpo_batchsize = 25000
#ant
if(which_agent==1):
	#training vars for new policy
	batchsize = 512
	nEpoch = 200
	learning_rate = 0.001
	#aggregation for training of new policy
	num_agg_iters = 5
	num_rollouts_to_agg= 5
	num_rollouts_testperformance = 3
	start_using_noised_actions = 50
	#other
	do_trpo = True

##########################################
##########################################

#get vars from saved MB run
param_dict = np.load('run_'+ str(run_num) + '/params.pkl')
N = param_dict['num_control_samples']
horizon = param_dict['horizon']
num_fc_layers_old = param_dict['num_fc_layers']
depth_fc_layers_old = param_dict['depth_fc_layers']
lr_olddynmodel = param_dict['lr']
batchsize_olddynmodel = param_dict['batchsize']
dt_steps = param_dict['dt_steps']
steps_per_rollout = param_dict['steps_per_episode']
tf_datatype = param_dict['tf_datatype']
seed = param_dict['seed']
if(tf_datatype=="<dtype: 'float64'>"):
	tf_datatype = tf.float64
else:
	tf_datatype = tf.float32

#load the saved MPC rollouts
f = open('run_'+ str(run_num)+'/savedRollouts.save', 'rb')
allData = cPickle.load(f)
f.close()

##########################################
##########################################

#create env
env, dt_from_xml = create_env(which_agent)

# set tf seed
npr.seed(seed)
tf.set_random_seed(seed)

#init vars
noise_onpol_rollouts=0.005
plot=False
print_frequency = 20 
validation_frequency = 50
num_fc_layers=2
depth_fc_layers=64
save_dir = 'run_'+ str(run_num)+'/mbmf'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#convert saved rollouts into array
allDataArray=[]
allControlsArray=[]
for i in range(len(allData)):
	allDataArray.append(allData[i]['observations'])
	allControlsArray.append(allData[i]['actions'])
training_data=np.concatenate(allDataArray)
labels=np.concatenate(allControlsArray)

if(len(labels.shape)==3):
	labels=np.squeeze(labels)
print("\n(total) Data size ", training_data.shape[0],"\n\n")

##################################################################################

# set aside some of the training data for validation
validnum = 10000
if((which_agent==6)or(which_agent==2)or(which_agent==1)):
	validnum=700
num = training_data.shape[0]-validnum
validation_x = training_data[num:num+validnum,:]
training_data=training_data[0:num,:]
validation_z = labels[num:num+validnum,:]
labels=labels[0:num,:]
print("\nTraining data size ", training_data.shape[0])
print("Validation data size ", validation_x.shape[0],"\n")

if(args.might_render or args.visualize_mlp_policy or args.visualize_on_policy_rollouts):
	might_render=True
else:
	might_render=False
#this somehow prevents a seg fault from happening in the later visualization
if(might_render):
    new_env = copy.deepcopy(env)
    new_env.render()

#gpu options for tensorflow
gpu_device = 0
gpu_frac = 0.3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
config = tf.ConfigProto(gpu_options=gpu_options,
                        log_device_placement=False,
                        allow_soft_placement=True,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)

#add SL noise to training data inputs and outputs
'''TO DO'''

#keep track of sample complexity
datapoints_used_forMB = np.load('run_'+ str(run_num) + '/datapoints_MB.npy')[-1]
datapoints_used_to_init_imit = training_data.shape[0]
total_datapoints = datapoints_used_forMB + datapoints_used_to_init_imit #points used thus far
imit_list_num_datapoints = []
imit_list_avg_rew = []

with tf.Session(config=config) as sess:

	if(not(args.use_existing_pretrained_policy)):

		#init vars
		g=GetTrueAction()
		g.make_model(sess, env, 'run_'+ str(run_num), tf_datatype, num_fc_layers_old, depth_fc_layers_old, which_agent, 
					lr_olddynmodel, batchsize_olddynmodel, N, horizon, steps_per_rollout, dt_steps, print_minimal)
		nData=training_data.shape[0]
		inputSize = training_data.shape[1]
		outputSize = labels.shape[1]

		#placeholders
		inputs_placeholder = tf.placeholder(tf_datatype, shape=[None, inputSize], name='inputs')
		labels_placeholder = tf.placeholder(tf_datatype, shape=[None, outputSize], name='outputs')

		#output of nn
		curr_output = nn_policy(inputs_placeholder, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype)

		#define training
		theta = tf.trainable_variables()
		loss = tf.reduce_mean(tf.square(curr_output - labels_placeholder))
		opt = tf.train.AdamOptimizer(learning_rate)
		gv = [(g,v) for g,v in opt.compute_gradients(loss, theta) if g is not None]
		train_step = opt.apply_gradients(gv)

		#get all the uninitialized variables (ie right now all of them)
		list_vars=[]
		for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
			if(not(tf.is_variable_initialized(var).eval())):
				list_vars.append(var)
		sess.run(tf.variables_initializer(list_vars))

		#aggregation iterations
		for agg_iter in range(num_agg_iters):

			print("ON AGGREGATION ITERATION ", agg_iter)
			rewards_for_this_iter=[]
			plot_trainingloss_x=[]
			plot_trainingloss_y=[]
			plot_validloss_x=[]
			plot_validloss_y=[]
			
			for i in range(nEpoch):

				################################
				############ TRAIN #############
				################################

				avg_loss=0
				iters_in_batch=0
				range_of_indeces = np.arange(training_data.shape[0])
				indeces = npr.choice(range_of_indeces, size=(training_data.shape[0],), replace=False)

				for batch in range(int(math.floor(nData / batchsize))):
					# Batch the training data
					inputs = training_data[indeces[batch*batchsize:(batch+1)*batchsize], :]
					outputs = labels[indeces[batch*batchsize:(batch+1)*batchsize], :]

					#one iteration of feedforward training
					_, my_loss = sess.run([train_step, loss], 
								feed_dict={inputs_placeholder: inputs, labels_placeholder: outputs})

					#loss
					avg_loss+= np.sqrt(my_loss)
					iters_in_batch+=1

				################################
				###### SAVE TRAIN LOSSES #######
				################################

				if(iters_in_batch==0):
					iters_in_batch=1

				current_loss = avg_loss/iters_in_batch

				#save training losses
				if(not(print_minimal)):
					if(i%print_frequency==0):
						print("training loss: ", current_loss, ", 	nEpoch: ", i)
				plot_trainingloss_x.append(i)
				plot_trainingloss_y.append(current_loss)
				np.save(save_dir + '/plot_trainingloss_x.npy', plot_trainingloss_x)
				np.save(save_dir + '/plot_trainingloss_y.npy', plot_trainingloss_y)

				################################
				########## VALIDATION ##########
				################################

				if((i%validation_frequency)==0):
					avg_valid_loss=0
					iters_in_valid=0

					range_of_indeces = np.arange(validation_x.shape[0])
					indeces = npr.choice(range_of_indeces, size=(validation_x.shape[0],), replace=False)

					for batch in range(int(math.floor(validation_x.shape[0] / batchsize))):
						# Batch the training data
						inputs = validation_x[indeces[batch*batchsize:(batch+1)*batchsize], :]
						outputs = validation_z[indeces[batch*batchsize:(batch+1)*batchsize], :]

						#one iteration of feedforward training
						my_loss, _ = sess.run([loss, curr_output], 
									feed_dict={inputs_placeholder: inputs, labels_placeholder: outputs})

						#loss
						avg_valid_loss+= np.sqrt(my_loss)
						iters_in_valid+=1

					curr_valid_loss = avg_valid_loss/iters_in_valid

					#save validation losses
					plot_validloss_x.append(i)
					plot_validloss_y.append(curr_valid_loss)
					if(not(print_minimal)):
						print("validation loss: ", curr_valid_loss, ", 	nEpoch: ", i, "\n")
					np.save(save_dir + '/plot_validloss_x.npy', plot_validloss_x)
					np.save(save_dir + '/plot_validloss_y.npy', plot_validloss_y)

			print("DONE TRAINING.")
			print("final training loss: ", current_loss, ", 	nEpoch: ", i)
			print("final validation loss: ", curr_valid_loss, ", 	nEpoch: ", i)

			##################
			##### PLOT #######
			##################
			if(plot):
				plt.plot(plot_validloss_x, plot_validloss_y, 'r')
				plt.plot(plot_trainingloss_x, plot_trainingloss_y, 'g')
				plt.show()

			##################################################
			##### RUN ON-POLICY ROLLOUTS --- DAGGER ##########
			##################################################

			print("\n\nCollecting on-policy rollouts...\n\n")
			starting_states = []
			observations = []
			actions=[]
			true_actions=[]

			for rollout in range(num_rollouts_to_agg):
				if(not(print_minimal)):
					print("\nOn rollout #", rollout)
				total_rew = 0

				starting_observation, starting_state = env.reset(returnStartState=True)
				curr_ob=np.copy(starting_observation)

				observations_for_rollout = []
				actions_for_rollout = []
				true_actions_for_rollout=[]
				for step in range(steps_per_rollout):
					
					#get action
					action = sess.run([curr_output], feed_dict={inputs_placeholder: np.expand_dims(curr_ob, axis=0)})
					action=np.copy(action[0][0]) #1x8

					#### add exploration noise to the action
					if(agg_iter>start_using_noised_actions):
						action = action + noise_onpol_rollouts*npr.normal(size=action.shape)

					#save obs and ac
					observations_for_rollout.append(curr_ob)
					actions_for_rollout.append(action)

					#####################################
					##### GET LABEL OF TRUE ACTION ######
					#####################################

					true_action = g.get_action(curr_ob)
					true_actions_for_rollout.append(true_action)

					#take step
					next_ob, rew, done, _ = env.step(action, collectingInitialData=False)
					total_rew+= rew
					curr_ob= np.copy(next_ob)

					if(done):
						break

					if((step%100)==0):
						print("   Done with step #: ", step)

				total_datapoints+= step
				print("rollout ", rollout," .... reward = ", total_rew)
				if(not(print_minimal)):
					print("number of steps: ", step)
					print("number of steps so far: ", total_datapoints)

				if(visualize_on_policy_rollouts):
					input("\n\nPAUSE BEFORE VISUALIZATION... Press Enter to continue...")
					visualize_rendering(starting_state, actions_for_rollout, env, dt_steps, dt_from_xml, which_agent)
					
				starting_states.append(starting_state)
				observations.append(observations_for_rollout)
				actions.append(actions_for_rollout)
				true_actions.append(true_actions_for_rollout)

				rewards_for_this_iter.append(total_rew)

			print("Avg reward for this iter: ", np.mean(rewards_for_this_iter), "\n\n")

			##################################################
			##### RUN CLEAN ROLLOUTS TO SEE PERFORMANCE ######
			##################################################

			print("\n\nTEST DAGGER PERFORMANCE (clean rollouts)...")
			rewards_for_this_iter2=[]
			for rollout in range(num_rollouts_testperformance):
				total_rew = 0
				starting_observation, starting_state = env.reset(returnStartState=True)
				curr_ob=np.copy(starting_observation)

				for step in range(steps_per_rollout):
					
					#get action
					action = sess.run([curr_output], feed_dict={inputs_placeholder: np.expand_dims(curr_ob, axis=0)})
					action=np.copy(action[0][0]) #1x8

					#take step
					next_ob, rew, done, _ = env.step(action, collectingInitialData=False)
					total_rew+= rew
					curr_ob= np.copy(next_ob)

					if(done):
						break
				if(not(print_minimal)):
					print("reward = ", total_rew)
				rewards_for_this_iter2.append(total_rew)
			print("Avg DAGGER performance at this iter: ", np.mean(rewards_for_this_iter2), "\n\n")

			###### SAVE datapoints vs performance
			imit_list_num_datapoints.append(total_datapoints)
			imit_list_avg_rew.append(total_rew)

			###########################
			##### AGGREGATE DATA ######
			###########################
			if(not(print_minimal)):
				print("\nAggregating Data...\n")
			training_data = np.concatenate([training_data, np.concatenate(observations)], axis=0)
			labels = np.concatenate([labels, np.concatenate(true_actions)], axis=0)

		#save the datapoints vs performance
		np.save('run_'+ str(run_num) + '/datapoints_IMIT.npy', imit_list_num_datapoints)
		np.save('run_'+ str(run_num) + '/performance_IMIT.npy', imit_list_avg_rew)

		if(not(print_minimal)):
			print("Done training the TF policy")

		######################
		### SAVE NN PARAMS ###
		######################

		#prepare the params for saving
		values = []
		for t in list_vars[0:6]:
			if(t.eval().shape==()):
				junk=1
			else:
				values.append(np.ndarray.flatten(t.eval()))
		values = np.concatenate(values)

		#save the TF policy params
		if(not(print_minimal)):
			print("Saving learned TF nn model parameters.")
		f = open(save_dir + '/policy_tf_values.save', 'wb')
		cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	else: #use_existing_pretrained_policy is True

		f = open(save_dir + '/policy_tf_values.save', 'rb')
		values = cPickle.load(f)
		f.close()

	#######################
	### INIT MLP POLICY ###
	#######################

	policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(depth_fc_layers, depth_fc_layers), init_std=std_on_mlp_policy)

	#copy params over to the MLP policy
	all_params = np.concatenate((values, policy._l_log_std.get_params()[0].get_value()))
	policy.set_param_values(all_params)

	#save the MLP policy
	f = open(save_dir + '/policy_mlp.save', 'wb')
	cPickle.dump(policy, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	if(not(print_minimal)):
		print("Done initializing MLP policy with a pre-trained policy.")

	##see what this initialized MLP policy looks like
	if(visualize_mlp_policy):
		input("\n\nPAUSE BEFORE VISUALIZATION... Press Enter to continue...")
	states, controls, starting_states, rewards = perform_rollouts(policy, 1, steps_per_rollout, visualize_mlp_policy, 
																CollectSamples, env, which_agent, dt_steps, dt_from_xml, False)
	print("Std of the MLP policy: ", std_on_mlp_policy)
	print("Reward of the MLP policy: ", rewards)
	
	################################
	### TRAIN MLP POLICY W/ TRPO ###
	################################

	if(do_trpo):
		run_experiment_lite(run_task, plot=True, snapshot_mode="all", use_cloudpickle=True, n_parallel=str(args.num_workers_trpo), 
						exp_name='run_' + str(run_num)+'_std' + str(std_on_mlp_policy)+ '_run'+ str(save_trpo_run_num),
						variant=dict(policy_values=values.tolist(), which_agent=which_agent, 
								trpo_batchsize=trpo_batchsize, steps_per_rollout=steps_per_rollout, 
								FiniteDifferenceHvp=FiniteDifferenceHvp, ConjugateGradientOptimizer=ConjugateGradientOptimizer, 
								depth_fc_layers=depth_fc_layers, std_on_mlp_policy=std_on_mlp_policy))
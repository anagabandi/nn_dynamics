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
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from rllab.misc.instrument import run_experiment_lite


def run_task(v):

	env, _ = create_env(v["which_agent"])
	policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))
	baseline = LinearFeatureBaseline(env_spec=env.spec)
	optimizer_params = dict(base_eps=1e-5)

	algo = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		batch_size=v["batch_size"],
		max_path_length=v["steps_per_rollout"],
		n_itr=v["num_trpo_iters"],
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
parser.add_argument('--seed', type=int, default='0')
parser.add_argument('--steps_per_rollout', type=int, default='1000')
parser.add_argument('--save_trpo_run_num', type=int, default='1')
parser.add_argument('--which_agent', type=int, default= 2)
parser.add_argument('--num_workers_trpo', type=int, default=2)
args = parser.parse_args()

batch_size = 50000

steps_per_rollout = args.steps_per_rollout
num_trpo_iters = 2500
if(args.which_agent==1):
	num_trpo_iters = 2500
if(args.which_agent==2):
	steps_per_rollout=333
	num_trpo_iters = 500
if(args.which_agent==4):
	num_trpo_iters= 2500
if(args.which_agent==6):
	num_trpo_iters= 2000

##########################################
##########################################

# set tf seed
npr.seed(args.seed)
tf.set_random_seed(args.seed)

run_experiment_lite(run_task, plot=True, snapshot_mode="all", use_cloudpickle=True, 
					n_parallel=str(args.num_workers_trpo), 
					exp_name='agent_'+ str(args.which_agent)+'_seed_'+str(args.seed)+'_mf'+ '_run'+ str(args.save_trpo_run_num),
					variant=dict(batch_size=batch_size, which_agent=args.which_agent, 
								steps_per_rollout=steps_per_rollout, num_trpo_iters=num_trpo_iters,
								FiniteDifferenceHvp=FiniteDifferenceHvp, ConjugateGradientOptimizer=ConjugateGradientOptimizer))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import genfromtxt
import joblib
import pandas
import argparse


######################
## ARGUMENTS TO SPECIFY
######################

parser = argparse.ArgumentParser()
parser.add_argument('--run_nums', type=int, nargs='+', default=-5)
parser.add_argument('--seeds', type=int, nargs='+', default=-5)
parser.add_argument('--which_agent', type=int, default=1)
parser.add_argument('--std_on_mlp_policy', type=float, default=0.5)
parser.add_argument('--batchsize_TRPO_mf', type=int, default=50000)
parser.add_argument('--batchsize_TRPO_mbmf', type=int, default=50000)
parser.add_argument('--dont_include_mbmfTRPO', action="store_true", dest='dont_include_mbmfTRPO', default=False)
parser.add_argument('--trpo_dir', type=str, default='/home/anagabandi/rllab/data/local/experiment/')
args = parser.parse_args()

######################
## vars
######################

#save args
which_agent = args.which_agent
std_on_mlp_policy = args.std_on_mlp_policy
batchsize_TRPO_mf = args.batchsize_TRPO_mf
batchsize_TRPO_mbmf = args.batchsize_TRPO_mbmf

#agent name
if(which_agent==2):
	agent_name='Swimmer'
if(which_agent==4):
	agent_name='Cheetah'
if(which_agent==6):
	agent_name='Hopper'
	batchsize_TRPO_mbmf= 25000
if(which_agent==1):
	agent_name='Ant'

#plotting vars
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
sns.set(font_scale=1)
format = 'png'
dpi=200

############################
## vars that depend on number of runs
############################

#seeds
how_many_seeds= len(args.seeds)

#run numbers for MB and imitation learning data
run_num1 = args.run_nums[0]
run_num2 = args.run_nums[0]
run_num3 = args.run_nums[0]
if(how_many_seeds==2):
	run_num1 = args.run_nums[0]
	run_num2 = args.run_nums[1]
	run_num3 = args.run_nums[1]
if(how_many_seeds==3):
	run_num1 = args.run_nums[0]
	run_num2 = args.run_nums[1]
	run_num3 = args.run_nums[2]

#filenames for MBMF TRPO
mbmf_filename_numbers = [1,1,1]
if(how_many_seeds==2):
	mbmf_filename_numbers = [1,2,2]
if(how_many_seeds==3):
	mbmf_filename_numbers = [1,2,3]

#filenames for MF TRPO
mf_filename_numbers = ['_seed_'+str(args.seeds[0])+'_mf_run1','_seed_'+str(args.seeds[0])+'_mf_run1','_seed_'+str(args.seeds[0])+'_mf_run1']
if(how_many_seeds==2):
	mf_filename_numbers = ['_seed_'+str(args.seeds[0])+'_mf_run1','_seed_'+str(args.seeds[1])+'_mf_run2','_seed_'+str(args.seeds[1])+'_mf_run2']
if(how_many_seeds==3):
	mf_filename_numbers = ['_seed_'+str(args.seeds[0])+'_mf_run1','_seed_'+str(args.seeds[1])+'_mf_run2','_seed_'+str(args.seeds[2])+'_mf_run3']

######################
## load in data
######################

#TRPO filenames to load in
pathname_mbmf1 = trpo_dir + 'run_'+ str(run_num1)+'_std'+str(std_on_mlp_policy) + '_run' +str(mbmf_filename_numbers[0])
pathname_mbmf2 = trpo_dir + 'run_'+ str(run_num2)+'_std'+str(std_on_mlp_policy) + '_run' +str(mbmf_filename_numbers[1])
pathname_mbmf3 = trpo_dir + 'run_'+ str(run_num3)+'_std'+str(std_on_mlp_policy) + '_run' +str(mbmf_filename_numbers[2])

#mf trpo runs
pathname_mf1 = trpo_dir + 'agent_'+str(which_agent)+ mf_filename_numbers[0]
pathname_mf2 = trpo_dir + 'agent_'+str(which_agent)+ mf_filename_numbers[1]
pathname_mf3 = trpo_dir + 'agent_'+str(which_agent)+ mf_filename_numbers[2]

#load in MB
MB_list_num_datapoints_run1 = np.load('../run_'+ str(run_num1) + '/datapoints_MB.npy')
MB_list_avg_rew_run1 = np.load('../run_'+ str(run_num1) + '/performance_MB.npy')
MB_list_num_datapoints_run2 = np.load('../run_'+ str(run_num2) + '/datapoints_MB.npy')
MB_list_avg_rew_run2 = np.load('../run_'+ str(run_num2) + '/performance_MB.npy')
MB_list_num_datapoints_run3 = np.load('../run_'+ str(run_num3) + '/datapoints_MB.npy')
MB_list_avg_rew_run3 = np.load('../run_'+ str(run_num3) + '/performance_MB.npy')

#load in imitation
imit_list_num_datapoints_run1 = np.load('../run_'+ str(run_num1) + '/datapoints_IMIT.npy')
imit_list_avg_rew_run1 = np.load('../run_'+ str(run_num1) + '/performance_IMIT.npy')
imit_list_num_datapoints_run2 = np.load('../run_'+ str(run_num2) + '/datapoints_IMIT.npy')
imit_list_avg_rew_run2 = np.load('../run_'+ str(run_num2) + '/performance_IMIT.npy')
imit_list_num_datapoints_run3 = np.load('../run_'+ str(run_num3) + '/datapoints_IMIT.npy')
imit_list_avg_rew_run3 = np.load('../run_'+ str(run_num3) + '/performance_IMIT.npy')

######################
## MB
######################

#performance
mb_run1= MB_list_avg_rew_run1[:6]
mb_run2= MB_list_avg_rew_run2[:6]
mb_run3= MB_list_avg_rew_run3[:6]

#datapoints
mb_num_data = MB_list_num_datapoints_run1[:6]

#mean and std of performance
mb_y = np.array([mb_run1, mb_run2, mb_run3])
mb_mean = mb_y.mean(axis=0)
mb_std = mb_y.std(axis=0)


######################
## MBMF
######################

if(args.dont_include_mbmfTRPO):
	#performance
	mbmf_run1 = np.concatenate([mb_run1, imit_list_avg_rew_run1])
	mbmf_run2 = np.concatenate([mb_run2, imit_list_avg_rew_run2])
	mbmf_run3 = np.concatenate([mb_run3, imit_list_avg_rew_run3])

	#datapoints
	mbmf_num_data = np.concatenate([mb_num_data, imit_list_num_datapoints_run1])

	#mean and std of performance
	mbmf_y = np.array([mbmf_run1, mbmf_run2, mbmf_run3])
	mbmf_mean = mbmf_y.mean(axis=0)
	mbmf_std = mbmf_y.std(axis=0)
else:
	#performance
	mbmf_run1_orig = np.array(pandas.read_csv(pathname_mbmf1+'/progress.csv')['AverageReturn'])
	mbmf_run2_orig = np.array(pandas.read_csv(pathname_mbmf2+'/progress.csv')['AverageReturn'])
	mbmf_run3_orig = np.array(pandas.read_csv(pathname_mbmf3+'/progress.csv')['AverageReturn'])

	mbmf_cutoff= np.min([mbmf_run1_orig.shape, mbmf_run2_orig.shape, mbmf_run3_orig.shape]) #make them all the same (min) length
	mbmf_run1_orig = mbmf_run1_orig[:mbmf_cutoff]
	mbmf_run2_orig = mbmf_run2_orig[:mbmf_cutoff]
	mbmf_run3_orig = mbmf_run3_orig[:mbmf_cutoff]

	mbmf_run1 = np.concatenate([mb_run1, imit_list_avg_rew_run1, mbmf_run1_orig])
	mbmf_run2 = np.concatenate([mb_run2, imit_list_avg_rew_run2, mbmf_run2_orig])
	mbmf_run3 = np.concatenate([mb_run3, imit_list_avg_rew_run3, mbmf_run3_orig])

	#datapoints
	datapoints_used_thus_far = imit_list_num_datapoints_run1[-1]
	mbmf_num_data_orig = batchsize_TRPO_mbmf*np.arange(mbmf_run1_orig.shape[0]+1)[1:] + datapoints_used_thus_far
	mbmf_num_data = np.concatenate([mb_num_data, imit_list_num_datapoints_run1, mbmf_num_data_orig])

	#mean and std of performance
	mbmf_y = np.array([mbmf_run1, mbmf_run2, mbmf_run3])
	mbmf_mean = mbmf_y.mean(axis=0)
	mbmf_std = mbmf_y.std(axis=0)

	print("MB datapoints: ", mb_num_data)
	print("MBMF datapoints: ", imit_list_num_datapoints_run1)

######################
## MF
######################

#performance

mf_run1 = pandas.read_csv(pathname_mf1+'/progress.csv')['AverageReturn']
mf_run2 = pandas.read_csv(pathname_mf2+'/progress.csv')['AverageReturn']
mf_run3 = pandas.read_csv(pathname_mf3+'/progress.csv')['AverageReturn']

mf_cutoff = np.min([mf_run1.shape, mf_run2.shape, mf_run3.shape]) #make them all the same (min) length
mf_run1=mf_run1[:mf_cutoff]
mf_run2=mf_run2[:mf_cutoff]
mf_run3=mf_run3[:mf_cutoff]

#datapoints
mf_num_data = batchsize_TRPO_mf*np.arange(mf_run1.shape[0]+1)[1:]

#mean and std of performance
mf_y = np.array([mf_run1, mf_run2, mf_run3])
mf_mean = mf_y.mean(axis=0)
mf_std = mf_y.std(axis=0)

######################
## PLOT
######################

fig, ax = plt.subplots(figsize=(7,3))

if(mb_num_data.shape[0]==1):
	ax.plot([mb_num_data[0],mb_num_data[0]], [0, mb_mean[0]], linewidth=2, color='g', label='Mb')
else:
	ax.plot(mb_num_data, mb_mean, color='g', label='Mb')
	ax.fill_between(mb_num_data, mb_mean - mb_std, mb_mean + mb_std, color='g', alpha=0.25)

ax.plot(mf_num_data, mf_mean, color='b', label='Mf')
ax.fill_between(mf_num_data, mf_mean - mf_std, mf_mean + mf_std, color='b', alpha=0.25)

ax.plot(mbmf_num_data, mbmf_mean, color='r', label='Mb-Mf (ours)', linewidth=0.5)
ax.fill_between(mbmf_num_data, mbmf_mean - mbmf_std, mbmf_mean + mbmf_std, color='r', alpha=0.25)

ax.hlines(mf_mean.max(), np.min([mb_num_data[0],mf_num_data[0]]), mf_num_data[-1], color='k', linestyle='--')

ax.semilogx()
ax.grid(True,which="both",ls="-")
ax.set_xlabel('Steps')
ax.set_ylabel('Cumulative Reward')
ax.set_title(agent_name)

ax.legend(loc='lower right')
fig.savefig(agent_name+'_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
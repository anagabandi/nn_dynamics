#!/bin/bash

#####################################
## SET VARS
#####################################

#location of all saved trpo runs
trpo_dir='/home/anagabandi/rllab/data/local/experiment/'

#specific to the run
how_many_seeds=3
seeds=(0 70 100)
num_workers_trpo=2 #how many cores to use

#specific to the agent
which_agent=4
std_on_mlp_policy=0.5
base_run_num=11 #used for filenames for saving

#####################################
## DO THE RUNS
#####################################

cd ..
echo 'run numbers:' 
iter_num=0
while [ $iter_num -lt $how_many_seeds ]
do
	seed=${seeds[$iter_num]}
	run_num=$(( $base_run_num + $iter_num ))
	echo $run_num
	save_trpo_run_num=$(( 1 + $iter_num ))

	python main.py --seed=$seed --run_num=$run_num --yaml_file='cheetah_forward'
	python mbmf.py --run_num=$run_num --which_agent=$which_agent --std_on_mlp_policy=$std_on_mlp_policy
	python trpo_run_mf.py --seed=$seed --save_trpo_run_num=$save_trpo_run_num --which_agent=$which_agent --num_workers_trpo=$num_workers_trpo

	iter_num=$(( $iter_num + 1))
done

#####################################
## PLOTTING
#####################################

cd plotting

if [ $how_many_seeds -eq 3 ]
then
python plot_mbmf.py --trpo_dir=$trpo_dir --std_on_mlp_policy=$std_on_mlp_policy --which_agent=$which_agent --run_nums 11 12 13 --seeds ${seeds[0]} ${seeds[1]} ${seeds[2]}
fi

if [ $how_many_seeds -eq 2 ]
then
python plot_mbmf.py --trpo_dir=$trpo_dir --std_on_mlp_policy=$std_on_mlp_policy --which_agent=$which_agent --run_nums 11 12 --seeds ${seeds[0]} ${seeds[1]}
fi

if [ $how_many_seeds -eq 1 ]
then
python plot_mbmf.py --trpo_dir=$trpo_dir --std_on_mlp_policy=$std_on_mlp_policy --which_agent=$which_agent --run_nums 11 --seeds ${seeds[0]}
fi

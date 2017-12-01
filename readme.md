# Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

[Arxiv Link](https://arxiv.org/abs/1708.02596)

**Abstract**: Model-free deep reinforcement learning algorithms have been shown to be capable of learning a wide range of robotic skills, but typically require a very large number of samples to achieve good performance. Model-based algorithms, in principle, can provide for much more efficient learning, but have proven difficult to extend to expressive, high-capacity models such as deep neural networks. In this work, we demonstrate that medium-sized neural network models can in fact be combined with model predictive control (MPC) to achieve excellent sample complexity in a model-based reinforcement learning algorithm, producing stable and plausible gaits to accomplish various complex locomotion tasks. We also propose using deep neural network dynamics models to initialize a model-free learner, in order to combine the sample efficiency of model-based approaches with the high task-specific performance of model-free methods. We empirically demonstrate on MuJoCo locomotion tasks that our pure model-based approach trained on just minutes of random action data can follow arbitrary trajectories, and that our hybrid algorithm can accelerate model-free learning on high-speed benchmark tasks, achieving sample efficiency gains of 3-5x on swimmer, cheetah, hopper, and ant agents. 
<!---
Videos can be found [here](https://sites.google.com/view/mbmf)
--> 

- For installation guide, go to [installation.md](https://github.com/nagaban2/learn_dynamics/blob/release/docs/installation.md)
- For notes on how to use your own environment, how to edit envs, etc. go to [notes.md](https://github.com/nagaban2/learn_dynamics/blob/release/docs/notes.md)

---------------------------------------------------------------

### How to run everything

```
cd scripts
./swimmer_mbmf.sh
./cheetah_mbmf.sh
./hopper_mbmf.sh
./ant_mbmf.sh
```

Each of those scripts does something similar to the following (but for multiple seeds):

```
python main.py --seed=0 --run_num=1 --yaml_file='swimmer_forward'
python mbmf.py --run_num=1 --which_agent=2
python trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=2 --num_workers_trpo=2 --std_on_mlp_policy=0.5
python plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=0.5 --which_agent=2 --run_nums 1 --seeds 0
```
Note that [trpo_dir] above corresponds to where the TRPO runs are saved. Probably somewhere in ~/rllab/data/... <br />
Each of these steps are further explained in the following sections.

---------------------------------------------------------------

### How to run MB

Need to specify:<br />

&nbsp;&nbsp;&nbsp;&nbsp;--**yaml_file** Specify the corresponding yaml file <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**seed** Set random seed to set for numpy and tensorflow <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**run_num** Specify what directory to save files under <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**use_existing_training_data** To use the data that already exists in the directory run_num instead of recollecting<br />
&nbsp;&nbsp;&nbsp;&nbsp;--**desired_traj_type** What type of trajectory to follow (if you want to follow a trajectory) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**num_rollouts_save_for_mf** Number of on-policy rollouts to save after last aggregation iteration, to be used later <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**might_render** If you might want to visualize anything during the run <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_MPC_rollout** To set a breakpoint and visualize the on-policy rollouts after each agg iteration <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**perform_forwardsim_for_vis** To visualize an open-loop prediction made by the learned dynamics model <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**print_minimal** To not print messages regarding progress/notes/etc. <br />

##### Examples:
```
python main.py --seed=0 --run_num=0 --yaml_file='cheetah_forward'
python main.py --seed=0 --run_num=1 --yaml_file='swimmer_forward'
python main.py --seed=0 --run_num=2 --yaml_file='ant_forward'
python main.py --seed=0 --run_num=3 --yaml_file='hopper_forward'
```
```
python main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='straight' --visualize_MPC_rollout
python main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='backward' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
python main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='forwardbackward' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
```
```
python main.py --seed=0 --run_num=5 --yaml_file='swimmer_trajfollow' --desired_traj_type='straight' --visualize_MPC_rollout
python main.py --seed=0 --run_num=5 --yaml_file='swimmer_trajfollow' --desired_traj_type='left_turn' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
python main.py --seed=0 --run_num=5 --yaml_file='swimmer_trajfollow' --desired_traj_type='right_turn' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
```
```
python main.py --seed=0 --run_num=6 --yaml_file='ant_trajfollow' --desired_traj_type='straight' --visualize_MPC_rollout
python main.py --seed=0 --run_num=6 --yaml_file='ant_trajfollow' --desired_traj_type='left_turn' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
python main.py --seed=0 --run_num=6 --yaml_file='ant_trajfollow' --desired_traj_type='right_turn' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
python main.py --seed=0 --run_num=6 --yaml_file='ant_trajfollow' --desired_traj_type='u_turn' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
```
---------------------------------------------------------------

### How to run MBMF

Need to specify:<br />

&nbsp;&nbsp;&nbsp;&nbsp;--**save_trpo_run_num number** Number used as part of directory name for saving mbmf TRPO run (you can use 1,2,3,etc to differentiate your different seeds) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**run_num** Specify what directory to get relevant MB data from & to save new MBMF files in <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**which_agent** Specify which agent (1 ant, 2 swimmer, 4 cheetah, 6 hopper) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**std_on_mlp_policy** Initial std you want to set on your pre-initialization policy for TRPO to use <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**num_workers_trpo** How many worker threads (cpu) for TRPO to use <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**might_render** If you might want to visualize anything during the run <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_mlp_policy** To visualize the rollout performed by trained policy (that will serve as pre-initialization for TRPO) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_on_policy_rollouts** To set a breakpoint and visualize the on-policy rollouts after each agg iteration of dagger <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**print_minimal** To not print messages regarding progress/notes/etc. <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**use_existing_pretrained_policy** To run only the TRPO part (if you've already done the imitation learning part in the same directory) <br />

*Note that the finished TRPO run saves to ~/rllab/data/local/experiments/*

##### Examples:
```
python mbmf.py --run_num=1 --which_agent=2 --std_on_mlp_policy=1.0
python mbmf.py --run_num=0 --which_agent=4 --std_on_mlp_policy=0.5
python mbmf.py --run_num=3 --which_agent=6 --std_on_mlp_policy=1.0 
python mbmf.py --run_num=2 --which_agent=1 --std_on_mlp_policy=0.5
```

---------------------------------------------------------------

### How to run MF

Run pure TRPO, for comparisons.<br /><br />

Need to specify command line args as desired<br />
&nbsp;&nbsp;&nbsp;&nbsp;--**seed** Set random seed to set for numpy and tensorflow <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**steps_per_rollout** Length of each rollout that TRPO should collect <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**save_trpo_run_num** Number used as part of directory name for saving TRPO run (you can use 1,2,3,etc to differentiate your different seeds) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**which_agent** Specify which agent (1 ant, 2 swimmer, 4 cheetah, 6 hopper) <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**num_workers_trpo** How many worker threads (cpu) for TRPO to use <br />
&nbsp;&nbsp;&nbsp;&nbsp;--**num_trpo_iters** Total number of TRPO iterations to run before stopping <br />

*Note that the finished TRPO run saves to ~/rllab/data/local/experiments/*


##### Examples:
```
python trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4 --num_workers_trpo=4
python trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=2 --num_workers_trpo=4
python trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=1 --num_workers_trpo=4
python trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=6 --num_workers_trpo=4

python trpo_run_mf.py --seed=50 --save_trpo_run_num=2 --which_agent=4 --num_workers_trpo=4
python trpo_run_mf.py --seed=50 --save_trpo_run_num=2 --which_agent=2 --num_workers_trpo=4
python trpo_run_mf.py --seed=50 --save_trpo_run_num=2 --which_agent=1 --num_workers_trpo=4
python trpo_run_mf.py --seed=50 --save_trpo_run_num=2 --which_agent=6 --num_workers_trpo=4
```
---------------------------------------------------------------

### How to plot

1) MBMF <br />
&nbsp;&nbsp;&nbsp;&nbsp;-Need to specify the commandline arguments as desired (in plot_mbmf.py) <br />
&nbsp;&nbsp;&nbsp;&nbsp;-Examples of running the plotting script: <br />
```
cd plotting
python plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=1.0 --which_agent=2 --run_nums 1 --seeds 0
python plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=1.0 --which_agent=2 --run_nums 1 2 3 --seeds 0 70 100
```
Note that [trpo_dir] above corresponds to where the TRPO runs are saved. Probably somewhere in ~/rllab/data/...

2) Dynamics model training and validation losses per aggregation iteration <br />
IPython notebook: plotting/plot_loss.ipynb <br />
Example plots: docs/sample_plots/... <br />

3) Visualize a forward simulation (an open-loop multi-step prediction of the elements of the state space) <br />
IPython notebook: plotting/plot_forwardsim.ipynb <br />
Example plots: docs/sample_plots/... <br />

4) Visualize the trajectories (on policy rollouts) per aggregation iteration <br />
IPython notebook: plotting/plot_trajfollow.ipynb <br />
Example plots: docs/sample_plots/... <br />


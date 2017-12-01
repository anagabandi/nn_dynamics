
# INSTALLING EVERYTHING

### ANACONDA (if you don't have)<br />

&nbsp;&nbsp;&nbsp;&nbsp;Download from https://www.continuum.io/downloads (download the python 2.7 version) <br />
```
bash Anaconda2-4.4.0-Linux-x86_64.sh
vim ~/.bashrc
```
In .bashrc, type:
```
export PATH="$HOME/anaconda2/bin:$PATH"
```
Source the file:
```
source ~/.bashrc
```

----------------------------------

### MUJOCO <br />

Go to website: https://www.roboti.us/license.html <br />

a) mujoco files:<br />
&nbsp;&nbsp;&nbsp;&nbsp;Under Downloads, download mjpro131 linux<br />
&nbsp;&nbsp;&nbsp;&nbsp;extract/unzip it <br />
	```
	mkdir ~/.mujoco
	cp -R mjpro131 ~/.mujoco/mjpro131
	```<br />
b) license key:<br />
&nbsp;&nbsp;&nbsp;&nbsp;i) If you don't have one: sign up for 30-day free trial to get a license<br />
&nbsp;&nbsp;&nbsp;&nbsp;Need to sudo chmod permissions on the downloaded executable (for getting computer id)<br />
&nbsp;&nbsp;&nbsp;&nbsp;Email will give you mjkey.txt + LICENSE.txt<br />
	```
	cp mjkey.txt ~/.mujoco/mjkey.txt
	``` <br />
&nbsp;&nbsp;&nbsp;&nbsp;ii) Else, just copy your existing key into ~/.mujoco/mjkey.txt
	
----------------------------------
	
### RLLAB

```
git clone https://github.com/nagaban2/rllab.git
cd rllab
./scripts/setup_linux.sh
./scripts/setup_mujoco.sh
vim ~/.bashrc
```
In .bashrc, type:
```
export PATH="$HOME/anaconda2/envs/rllab3/bin:$PATH"
export PYTHONPATH="$HOME/rllab:$PYTHONPATH"
```
Source the file:
```
source ~/.bashrc
source activate rllab3
```
----------------------------------

### CUDA (Note: assuming you already have cuda and cudnn)

&nbsp;&nbsp;&nbsp;&nbsp;Set paths:
```
vim ~/.bashrc
```
In .bashrc, type:
```
export PATH="/usr/local/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
```
Source the file:
```
source ~/.bashrc
```
&nbsp;&nbsp;&nbsp;&nbsp;To see if gpu is being used while running code:
```
nvidia-smi
```
----------------------------------
	
### OTHER
```
source activate rllab3
pip install gym
pip install cloudpickle
pip install seaborn
```
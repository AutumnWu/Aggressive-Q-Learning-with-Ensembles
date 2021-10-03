# Aggressive-Q-Learning-with-Ensembles AQE Pytorch Implementation
AQE Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This AQE implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. 
Currently anonymized for reviewing.

## Setup environment:
To use the code you should first download this repo, and then install spinup:

The spinup documentation is here, you should read it to make sure you know the procedure: https://spinningup.openai.com/en/latest/user/installation.html

The only difference in installation is you want to install this forked repo, instead of the original repo.

The Pytorch version used is: 1.3.1, install pytorch:
https://pytorch.org/

Mujoco_py version used is: 2.0.2.1

## Run experiment
The AQE implementation can be found under `spinup/algos/aqe/`

Run experiments with pytorch aqe:

In the aqe folder, run the AQE code with `python aqe.py`

AQE: set `num_Qs = 10, q_target_mode = 'aqe', multihead = 2`



You can run SAC/REDQ by changing the parameter `num_Qs` and `q_target_mode`


Run experiments with pytorch SAC:

SAC: set `num_Qs = 2, num_mins = 2, q_target_mode = 'min',  multihead = 1`



Run experiments with pytorch REDQ:

REDQ: set `num_Qs > 2, num_mins = 2, q_target_mode = 'min', multihead = 1`


Note: currently there is no parallel running for SAC and AQE (also not supported by spinup), so you should always set number of cpu to 1 when you use experiment grid.

The program structure, though in Pytorch has been made to be as close to spinup tensorflow code as possible so readers who are familiar with other algorithm code in spinup will find this one easier to work with. I also referenced rlkit's SAC pytorch implementation, especially for the policy and value models part, but did a lot of simplification. 

Consult Spinup documentation for output and plotting:

https://spinningup.openai.com/en/latest/user/saving_and_loading.html

https://spinningup.openai.com/en/latest/user/plotting.html


## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

The code will be released as a public github repo, with better documentation after the reviewing process. 

# Reinforcement learning on ATRP

This repo is presented to reproduce results in https://arxiv.org/abs/1712.04516.

## To setup
```
# build a virtualenv
virtualenv atrprl -p python3
source atrprl/bin/activate

# install requirements
pip install -r requirements.txt
```

## To run
- A demo of training to a target MWD of a Gaussian with variance 24, using 12 CPU cores by default and saving training results to `./output`:  
  `python sample_train.py atrpenv_gv.py ATRP-psnt-td-gv24-v0`

- A demo of testing on the above training:  
  `python sample_test.py atrpenv_gv.py ATRP-psnt-td-gv24-v0 ./output/ATRP-psnt-td-gv24-v0-run1/model_0.h5`
  
In the above line, file `model_0.h5` corresponds to an untrained RL agent, and so the code should run but the performance will be close to a random (untrained) agent.

## Requirements
```
numpy
scipy
gym
matplotlib
pygame
h5py
tensorflow
-e git+https://github.com/spring01/simatrp.git#egg=simatrp
-e git+https://github.com/spring01/drlbox.git#egg=drlbox
```

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

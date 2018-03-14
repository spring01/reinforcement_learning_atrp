'''
Example: python sample_train.py atrpenv_gv.py ATRP-psnt-td-gv24-v0
'''
import os; os.environ['OMP_NUM_THREADS'] = '1'
import sys
import gym
import importlib
import numpy as np
from tensorflow.python.keras.layers import Input, Conv1D, Flatten, Dense, Activation
from history import HistoryStacker
from obs_dmm import ObsDMM
from act_single import ActSingle
from drlbox.trainer import make_trainer


def main():
    trainer = make_trainer('a3c',
                           env_maker=lambda: make_env(*sys.argv[1:]),
                           feature_maker=lambda o: make_feature(o, num_hid=100),
                           state_to_input=state_to_input,
                           num_parallel=12,
                           train_steps=100000000,
                           interval_save=100000,
                           save_dir='output',
                           catch_signal=True,
                           verbose=True,)
    trainer.run()


def make_env(filename, envname):
    import_name, _ = os.path.splitext(filename)
    import_name = import_name.replace('/', '.')
    importlib.import_module(import_name)
    env = gym.make(envname).unwrapped
    env = ObsDMM(env)
    env = ActSingle(env)
    return HistoryStacker(env, num_frames=1, act_steps=4)

def state_to_input(state):
    return np.stack(state, axis=-1)

def make_feature(observation_space, num_hid):
    spaces = observation_space.spaces
    inp_state = Input(shape=(spaces[0].shape[0], len(spaces)))
    relu = Activation('relu')
    conv1 = Conv1D(8, 32, strides=2)(inp_state)
    conv1 = relu(conv1)
    conv2 = Conv1D(8, 32, strides=1)(conv1)
    conv2 = relu(conv2)
    flattened = Flatten()(conv2)
    feature = Dense(num_hid)(flattened)
    feature = relu(feature)
    return inp_state, feature


if __name__ == '__main__':
    main()


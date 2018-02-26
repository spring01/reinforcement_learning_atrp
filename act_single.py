
import gym
import numpy as np


'''
Action mode 'single':
    Only one addable may be added at a time.
'''
class ActSingle(gym.Wrapper):

    def __init__(self, env, addable_allowed=[True, True, True, True, True]):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(sum(addable_allowed) + 1)
        self.parse_action_dict = {0: (0, 0, 0, 0, 0)}
        single_addition_list = [(1, 0, 0, 0, 0),
                                (0, 1, 0, 0, 0),
                                (0, 0, 1, 0, 0),
                                (0, 0, 0, 1, 0),
                                (0, 0, 0, 0, 1)]
        key = 0
        for allowed, action in zip(addable_allowed, single_addition_list):
            if allowed:
                key += 1
                self.parse_action_dict[key] = action

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.parse_action_dict[action]
        return self.env.step(action)


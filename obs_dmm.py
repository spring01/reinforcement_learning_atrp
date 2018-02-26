
import gym
import numpy as np
from simatrp.atrp_base import MONO, CU1, CU2, DORM


'''
Observation mode 'dmm':
    "dormant-monomer-mix" mode; capped indicators, volume, and quantities of
    dormant chains and monomer mixed (dorm-1 and monomer quantities are summed
    together), cu1 and cu2.
'''
class ObsDMM(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_len = 5 + 1 + env.max_rad_len + 2
        self.observation_space = gym.spaces.Box(0.0, np.inf, shape=(obs_len,),
                                                dtype=np.float32)

    def reset(self):
        self.env.reset()
        return self.observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(), reward, done, info

    def observation(self):
        capped = [self.env.capped(key) for key in self.env.addables]
        obs_cap_vol = np.concatenate([capped, [self.env.volume]])
        quant = lambda key: self.env.quant[self.env.index[key]]
        dmm = quant(DORM).copy()
        dmm[0] += quant(MONO)
        obs_chains = [dmm, [quant(CU1), quant(CU2)]]
        obs_chains = np.concatenate(obs_chains)
        return np.concatenate([obs_cap_vol, obs_chains])


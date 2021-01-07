import time
import joblib
import os
import os.path as osp
import torch
from spinup_utils import *
import gym
import safety_gym
from safety_gym.envs.engine import Engine
from test_policy import *


def preprocess_buffer():
    pass

def generate_buffer_samples():
    pass


class BCO(nn.Module):
    def __init__(self, env, policy='mlp'):
        super(BCO, self).__init__()

        self.policy = policy
        self.act_n = env.action_space.shape[0]

        if self.policy == 'mlp': # MLP architecture
            self.obs_n = env.observation_space.shape[0]
            self.pol = nn.Sequential(*[nn.Linear(self.obs_n, 32), nn.LeakyReLU(),
                                       nn.Linear(32, 32), nn.LeakyReLU(),
                                       nn.Linear(32, self.act_n)])
            self.inv = nn.Sequential(*[nn.Linear(self.obs_n * 2, 32), nn.LeakyReLU(),
                                       nn.Linear(32, 32), nn.LeakyReLU(),
                                       nn.Linear(32, self.act_n)])
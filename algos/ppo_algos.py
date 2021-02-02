import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

from torch.nn import Parameter

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from utils import *


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        print("test list")
        print(list(hidden_sizes))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)


    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function critics
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)


    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
            # pen = self.pen(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):
        mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
        std = self.var_net(F.leaky_relu(self.shared_net(x)))
        return Normal(loc=mu, scale=std).rsample()



class DistilledGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, n_experts):
        super().__init__()
        obs_dim_aug = obs_dim + n_experts
        self.shared_net = mlp([obs_dim_aug] + list(hidden_sizes), activation)

        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):

        out = F.leaky_relu(self.shared_net(x))
        # print("out matrix")
        # print(out)
        # print("mu")
        mu = self.mu_net(out)
        # print(mu)
        # print("std out")
        std = self.var_net(out)
        # print(std)

        # mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
        # std = self.var_net(F.leaky_relu(self.shared_net(x)))
        return Normal(loc=mu, scale=std).rsample()
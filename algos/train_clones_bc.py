import sys, os, random, pickle

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.distributions.independent import Independent
from cpprb import ReplayBuffer, create_before_add_func

from collections import namedtuple
from torch.distributions import Normal

from adabelief_pytorch import AdaBelief
from ppo_algos import *

import gym
import wandb
import safety_gym

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'
DATA_PATH = '/home/tyna/Documents/openai/research-project/expert_data/'
config_name = 'cyan'
DEMO_DIR = os.path.join(DATA_PATH, config_name + '_episodes/')
hid_size = 128
RENDER = True
max_steps = 1000
episode_tests = 10
epochs = 100

# make environment
env = gym.make(ENV_NAME)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape


# PREPARE DATA
num_files = 50

data_all = None

f = open(DEMO_DIR + 'sim_data_2611_buffer.pkl', "rb" )
BufFile = pickle.load(f)
f.close()

print("Buf FILE")
# print(BufFile)
print('No.')


def raw_numpy(raw):
    # return raw: np.array and done: np.array
    _mask = torch.ones(len(raw), dtype=torch.bool) # Tyna change this mask eventually
    done= ~_mask
    return raw, done


class Samples:
    def __init__(self, states=None, actions=None, rewards=None,
                 next_states=None, weights=None, indexes=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.weights = weights
        self.indexes = indexes
        self._keys = [self.states, self.actions, self.rewards,
                      self.next_states, self.weights, self.indexes]

    def __iter__(self):
        return iter(self._keys)

def samples_to_np(samples):

    # np_dones = samples.states   # Tyna to fix, this is not correct
    np_states, np_dones = raw_numpy(samples.states)

    np_actions = samples.actions
    np_rewards = samples.rewards
    np_next_states, np_next_dones = samples.next_states
    return np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones

def samples_from_cpprb(npsamples, device=None):
    """
    Convert samples generated by cpprb.ReplayBuffer.sample() into
    State, Action, rewards, State.
    Return Samples object.
    Args:
        npsamples (dict of nparrays):
            Samples generated by cpprb.ReplayBuffer.sample()
        device (optional): The device where the outputs are loaded.
    Returns:
        Samples(State, Action, torch.FloatTensor, State)
    """
    # device = self.device if device is None else device

    # states = State.from_numpy(npsamples["obs"])
    # actions = Action.from_numpy(npsamples["act"])
    # rewards = torch.tensor(npsamples["rew"], dtype=torch.float32).squeeze()
    # next_states = State.from_numpy(npsamples["next_obs"], npsamples["done"])


    states = npsamples["obs"]
    actions = npsamples["act"]
    rewards = torch.tensor(npsamples["rew"], dtype=torch.float32).squeeze()
    next_states = npsamples["next_obs"], npsamples["done"]

    # if self.prioritized:
    #     weights = torch.tensor(
    #         npsamples["weights"], dtype=torch.float32, device=self.device)
    #     indexes = npsamples["indexes"]
    # else:
    #     weights = torch.ones(states.shape[0], device=self.device)
    #     indexes = None
    return Samples(states, actions, rewards, next_states)

print("buffer data")
data = samples_from_cpprb(npsamples=BufFile)
# print(data)

np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)

before_add = create_before_add_func(env)

replay_buffer = ReplayBuffer(size=1000,
                             env_dict={
                              "obs": {"shape": obs_dim},
                              "act": {"shape": act_dim},
                              "rew": {},
                              "next_obs": {"shape": obs_dim},
                              "done": {}})

print("about to add to replay buffer")
replay_buffer.add(**before_add(obs=np_states[~np_dones],
                       act=np_actions[~np_dones],
                       rew=np_rewards[~np_dones],
                       done=np_next_dones[~np_dones],
                       next_obs=np_next_states[~np_dones]))

# print("done adding to replay buffer")
#
# print(replay_buffer.sample(32))
# print("nailed some samples")

# print("hello!")
# print(np_states)
# data = BufFile.get()
# print(data)

# for i in range(num_files):
#     # print(i)
#     # Append data from files, chosen randomly
#     random_file = random.choice(os.listdir(DEMO_DIR))
#     # print(random_file)
#     fil = pd.read_pickle(DEMO_DIR + random_file)
#     # print(fil)
#     if data_all is None:
#         data_all = fil
#     else:
#         data_all = data_all.append(fil)


# Create policy class
class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, action_dim)
        self.linear_var = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.001)
        x_mu = self.linear_mu(x)
        x_var = self.linear_var(x)
        return x_mu, x_var


# Create expert demonstration class
# class ExpertDemos(Dataset):
#     def __init__(self, demos):
#         self.data = []
#         for i, row in enumerate(demos.values):
#             state, action, reward, cost, done = row
#             self.data.append((state, action, reward, cost, done))
#             # print("here is the data")
#         print("first action")
#         print(self.data[0][1])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         s, a, r, c, d = self.data[idx]
#         return s, a, r, c, d
#
# class PPO_Buffer:
#     def __init__(self, maxlen=100000, device=None):
#         self.mem = deque(maxlen=maxlen)
#         self.maxlen = maxlen
#         self.device = device
#
#     def store(self, s, a, r, s_, a_, d):
#         self.mem.append([s, a, r, s_, a_, d])
#
#     def sample(self, batch_size):
#         bat = random.sample(self.mem, batch_size)
#         batch = list(zip(*bat))
#         data = []
#         for i in range(len(batch)):
#             data.append(T.as_tensor(batch[i], dtype=T.float32, device=self.device))
#         return data
#
#     def __len__(self):
#         return len(self.mem)

# create dataset using demos class for the chosen dataset
# dataset = ExpertDemos(data_all)

# dataloader
BATCH_SIZE=100
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

pi = Policy(env.observation_space.shape[0], hid_size, env.action_space.shape[0])
pi_optimizer = Adam(pi.parameters(), lr=3e-4, weight_decay=0.0001)
criterion = nn.MSELoss()

# (states, actions, _, _, _, _) = self.replay_buffer.sample(
#                 self.minibatch_size)
#             policy_actions = Action(self.policy(states))
#             loss = mse_loss(policy_actions.features, actions.features)
#             self.policy.reinforce(loss)
#             self.writer.train_steps += 1


class RLNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """

    def __init__(self, model, _=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return self.model(state.features.float()) * state.mask.float().unsqueeze(-1)

    def to(self, device):
        self.device = device
        return super().to(device)


class GaussianPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]

    def forward(self, state, return_mean=False):
        outputs = super().forward(state)
        means = outputs[:, :self._action_dim]

        if return_mean:
            return means

        logvars = outputs[:, self._action_dim:]
        std = logvars.exp_()
        return Independent(Normal(means, std), 1)

    def to(self, device):
        return super().to(device)

for epoch in range(epochs):
    total_loss = 0
    # for i, data in enumerate(loader):
    for i in range(10):
        print(i)
        # s, a, r, c, d = data
        # print("Action! ", a)
        states, actions, _, _, _ = replay_buffer.sample(BATCH_SIZE)
        wow = replay_buffer.sample(BATCH_SIZE)
        print(wow['obs'])
        print(wow['obs'].shape)
        print("samples of states")
        # print(states)
        # print(actions)
        pi_optimizer.zero_grad()
        # Policy loss
        a_mu, a_sigma = pi(s.float())
        a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
        loss = criterion(a_pred, a)
        # print("Loss!", loss)
        total_loss += loss.item()
        loss.backward()
        if i % 20 == 19:
            print('Epoch:%d Batch:%d Loss:%.3f' % (epoch, i + 1, total_loss / 20))
            total_loss = 0
        pi_optimizer.step()

# for epoch in range(epochs):
#     total_loss = 0
#     for i, data in enumerate(loader):
#         s, a, r, c, d = data
#         # print("Action! ", a)
#         pi_optimizer.zero_grad()
#         # Policy loss
#         a_mu, a_sigma = pi(s.float())
#         a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
#         loss = criterion(a_pred, a)
#         # print("Loss!", loss)
#         total_loss += loss.item()
#         loss.backward()
#         if i % 20 == 19:
#             print('Epoch:%d Batch:%d Loss:%.3f' % (epoch, i + 1, total_loss / 20))
#             total_loss = 0
#         pi_optimizer.step()


# Play clone episodes
returns = []
for i in range(episode_tests):
    print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        a_mu, a_sigma = pi(torch.from_numpy(obs).float())
        a = Normal(loc=a_mu, scale=a_sigma).sample()
        obs, r, done, _ = env.step(a.detach().numpy())
        if RENDER:
            env.render()
        totalr += r
        steps += 1
        if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if steps >= max_steps:
            break
    returns.append(totalr)

print('Returns', returns)
print('Avg EpRet', np.mean(returns))
print('Std EpRet', np.std(returns))


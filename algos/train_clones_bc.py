import sys, os
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import torch.optim as optim




from adabelief_pytorch import AdaBelief
from ppo_algos import *

import pickle
import gym
import safety_gym

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'
DATA_PATH = '/home/tyna/Documents/openai/research-project/data/'
configname = 'cyan'
DEMO_DIR = os.path.join(DATA_PATH, configname + '_episodes/' )
print("demo dir:", DEMO_DIR)
RENDER = False

### PREPARE DATA
fil = pd.read_pickle(DEMO_DIR + 'sim_data_4083.pkl')
env = gym.make(ENV_NAME)


class policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(policy, self).__init__()
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


class expert_demos(Dataset):
    def __init__(self, demos):
        self.data = []
        for i, row in enumerate(demos.values):
            state, action, reward, cost, done  = row
            self.data.append((state, action, reward, cost, done))
            # print("here is the data")
        print("first action")
        print(self.data[0][1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, r, c, d = self.data[idx]
        return s, a, r, c, d

dataset = expert_demos(fil)

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# ac_kwargs=dict(hidden_sizes=[128] * 4)
# ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)

pi = policy(env.observation_space.shape[0], 32, env.action_space.shape[0])
pi_optimizer = Adam(pi.parameters(), lr=3e-4, weight_decay=0.0001)
criterion = nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for i, data in enumerate(loader):
        s, a, r, c, d = data
        # print("Action! ", a)
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


RENDER = True

max_steps = 1000
pi
returns = []
for i in range(10):
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



class BC:
    pass

if __name__ == '__main__':
    print("behavioral clones")
    # dataset = imitation_dataset(demos)
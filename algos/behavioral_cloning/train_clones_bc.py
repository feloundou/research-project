import sys, os, random, pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.distributions.independent import Independent
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict, train
from run_policy_sim_ppo import *

# from collections import namedtuple
# from torch.distributions import Normal
from adabelief_pytorch import AdaBelief

from utils import *
from ppo_algos import *


import gym
import wandb
import safety_gym

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'
DATA_PATH = '/home/tyna/Documents/openai/research-project/expert_data/'
base_path = '/home/tyna/Documents/openai/research-project/data/'
# expert_path = '/home/tyna/Documents/openai/research-project/expert_data/'
# config_name = 'cyan'
config_name = 'hyacinth'
DEMO_DIR = os.path.join(DATA_PATH, config_name + '_episodes/')

RENDER = True
MAX_STEPS = 1000
BATCH_SIZE = 100
replay_buffer_size = 10000

episode_tests = 10
epochs = 100
train_iters=100

# Neural Network Size
hid_size = 128
n_layers = 2
record=True

fname = config_name + "_clone_" + str(epochs) + 'ep_' + str(train_iters) + 'train_it'
seed=0

# Random seed
seed += 10000 * proc_id()
torch.manual_seed(seed)
np.random.seed(seed)

# make environment
env = gym.make(ENV_NAME)
env_dict = create_env_dict(env)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

# PREPARE DATA
# Should we get the replay buffer from file or not?
pull_from_file = False

if pull_from_file:
    f = open(DEMO_DIR + 'sim_data_7263_buffer.pkl', "rb")
    buffer_file = pickle.load(f)
    f.close()

    data = samples_from_cpprb(npsamples=buffer_file)

    # Reconstruct the data, then pass it to replay buffer
    np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)

    # Create environment
    before_add = create_before_add_func(env)

    replay_buffer = ReplayBuffer(size=replay_buffer_size,
                                 env_dict={
                                  "obs": {"shape": obs_dim},
                                  "act": {"shape": act_dim},
                                  "rew": {},
                                  "next_obs": {"shape": obs_dim},
                                  "done": {}})

    replay_buffer.add(**before_add(obs=np_states[~np_dones],
                                   act=np_actions[~np_dones],
                                   rew=np_rewards[~np_dones],
                                   next_obs=np_next_states[~np_dones],
                                   done=np_next_dones[~np_dones]))
else:
    # Generate expert data
    file_name = 'ppo_penalized_' + config_name + '_20Ks_1Ke_128x4'

    # Load  trained policy
    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
                                        'last',
                                        False)

    # Gather expert experiences from the trained policy
    expert_rb = run_policy(env,
                           get_action,
                           0,
                           episode_tests,
                           False,
                           record=record,
                           data_path=DATA_PATH,
                           config_name=config_name,
                           max_len_rb=replay_buffer_size)

    # buffer_file = expert_rb
    replay_buffer = expert_rb

# Create policy class


# class Policy(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(Policy, self).__init__()
#         self.linear_1 = nn.Linear(state_dim, hidden_dim)
#         self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear_mu = nn.Linear(hidden_dim, action_dim)
#         self.linear_var = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, x):
#         x = self.linear_1(x)
#         x = F.leaky_relu(x, 0.001)
#         x = self.linear_2(x)
#         x = F.leaky_relu(x, 0.001)
#         x_mu = self.linear_mu(x)
#         x_var = self.linear_var(x)
#         return x_mu, x_var

# Special function to avoid certain slowdowns from PyTorch + MPI combo.
setup_pytorch_for_mpi()

# Set up logger and save configuration
logger_kwargs = setup_logger_kwargs(fname, seed)
logger = EpochLogger(**logger_kwargs)
# logger.save_config(locals()) # configs for now are in my config file, might revisit this later

# Setup policy, optimizer and criterion
# clone_pi = Policy(env.observation_space.shape[0], hid_size, env.action_space.shape[0])
ac_kwargs=dict(hidden_sizes=[hid_size] * n_layers)
clone_pi = MLPGaussianActor(obs_dim, env.action_space.shape[0], activation=nn.Tanh, hidden_sizes=[128,128])
pi_optimizer = Adam(clone_pi.parameters(), lr=3e-4, weight_decay=0.0001)
criterion = nn.MSELoss()

# Sync params across process
sync_params(clone_pi)

# Train without environment interaction
wandb.login()
wandb.init(project='behavioral_clone_training', name=fname)
wandb.watch(clone_pi)  # watch neural net

for epoch in range(epochs):
    total_loss = 0
    # for i, data in enumerate(loader):
    for i in range(train_iters):

        # Sample from the replay buffer
        SAMPLE = replay_buffer.sample(BATCH_SIZE)

        # Observe states and chosen actions from expert
        # seems rewards and costs are not relevant here since clone will not receive them
        states = SAMPLE['obs']
        actions = SAMPLE['act']

        pi_optimizer.zero_grad()
        # Policy loss
        pi  = clone_pi._distribution(states)
        a = pi.sample()
        # logp_a = pi._log_prob_from_distribution(pi, a)
        # a_mu, a_sigma = clone_pi(torch.tensor(states).float())
        # a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
        a_pred= a
        print("predicted a: ", a)
        loss = criterion(a_pred, torch.tensor(actions))
        
        print("Loss!", loss)
        total_loss += loss.item()
        loss.backward()
        if i % 20 == 19:
            print('Epoch:%d Batch:%d Loss:%.3f' % (epoch, i + 1, total_loss / 20))
            total_loss = 0
            epoch_metrics = {'20it average epoch loss': total_loss / 20}
            wandb.log(epoch_metrics)
        pi_optimizer.step()

    # Once done, save the clone model
    logger.setup_pytorch_saver(clone_pi)

wandb.finish()

# Play episodes and record
if record:
    # Logging
    # wandb.login()
    wandb.init(project="clone_benchmarking_" + config_name , name=config_name + "_clone_" + str(epochs) + 'ep_' + str(train_iters) + 'train_it')

    rew_mov_avg_10 = []
    cost_mov_avg_10 = []

    returns = []
    costs = []

    cum_ret = 0
    cum_cost = 0

    # Play clone episodes
    for i in range(episode_tests):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        totalc = 0.
        steps = 0

        while not done:
            a_mu, a_sigma = pi(torch.from_numpy(obs).float())
            a = Normal(loc=a_mu, scale=a_sigma).sample()
            obs, r, done, info = env.step(a.detach().numpy())
            cost = info['cost']
            if RENDER:
                env.render()
            totalr += r
            totalc += cost
            steps += 1
            if steps % 100 == 0: print("%i/%i" % (steps, MAX_STEPS))
            if steps >= MAX_STEPS:
                break
        returns.append(totalr)
        costs.append(totalc)

        cum_ret += totalr
        cum_cost += totalc

        if len(rew_mov_avg_10) >= 25:
            rew_mov_avg_10.pop(0)
            cost_mov_avg_10.pop(0)

        rew_mov_avg_10.append(totalr)
        cost_mov_avg_10.append(totalc)

        mov_avg_ret = np.mean(rew_mov_avg_10)
        mov_avg_cost= np.mean(cost_mov_avg_10)

        clone_metrics = {'episode return': totalr,
                          'episode cost': totalc,
                         'cumulative return': cum_ret,
                         'cumulative cost': cum_cost,
                          '25ep mov avg return': mov_avg_ret,
                          '25ep mov avg cost': mov_avg_cost
                          }
        wandb.log(clone_metrics)

    wandb.finish()

    print('Returns', returns)
    print('Avg EpRet', np.mean(returns))
    print('Std EpRet', np.std(returns))
    print('Costs', costs)
    print('Avg EpCost', np.mean(costs))
    print('Std EpCost', np.std(costs))


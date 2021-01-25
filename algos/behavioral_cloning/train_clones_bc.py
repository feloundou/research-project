import sys, os, random, pickle
import numpy as np
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F
from six.moves.collections_abc import Sequence

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
config_name = 'lilly'
DEMO_DIR = os.path.join(DATA_PATH, config_name + '_episodes/')

RENDER = True
efficiency_eval= False
record_clone=False
record_expert=True
# PREPARE DATA
# Should we get the replay buffer from file or not?
pull_from_file = False

MAX_STEPS = 1000
save_every= 10

BATCH_SIZE = 100
replay_buffer_size = 10000

episode_tests = 50
record_tests = 100

epochs = 250
train_iters=100

# Neural Network Size
hid_size = 128
n_layers = 2


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



if pull_from_file:
    f = open(DEMO_DIR + 'sim_data_' + str(episode_tests) + '_buffer.pkl', "rb")
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
                           record=record_expert,
                           data_path=DATA_PATH,
                           config_name=config_name,
                           max_len_rb=replay_buffer_size)

    # buffer_file = expert_rb
    replay_buffer = expert_rb

# Create policy class

#
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



class GaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        print("obs vals")
        print([obs_dim] + list(hidden_sizes) + [act_dim])
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):
        mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
        std = self.var_net(F.leaky_relu(self.shared_net(x)))

        return Normal(loc=mu, scale=std).rsample()




# Special function to avoid certain slowdowns from PyTorch + MPI combo.
setup_pytorch_for_mpi()

# Set up logger and save configuration
logger_kwargs = setup_logger_kwargs(fname, seed)
logger = EpochLogger(**logger_kwargs)
# logger.save_config(locals()) # configs for now are in my config file, might revisit this later

# Setup policy, optimizer and criterion
# clone_pi = Policy(env.observation_space.shape[0], hid_size, env.action_space.shape[0])
ac_kwargs= dict(hidden_sizes=[hid_size] * n_layers)

clone_pi = GaussianActor(obs_dim[0], env.action_space.shape[0], activation=nn.LeakyReLU, **ac_kwargs)
# pi_optimizer = Adam(clone_pi.parameters(), lr=3e-4, weight_decay=0.0001)
pi_optimizer = AdaBelief(clone_pi.parameters(), betas=(0.9, 0.999), eps=1e-16)

criterion = nn.MSELoss()

# Once done, save the clone model
logger.setup_pytorch_saver(clone_pi)


# Sync params across process
sync_params(clone_pi)

# Train without environment interaction
wandb.login()
wandb.init(project='behavioral_clone_training', name=fname)
wandb.watch(clone_pi)  # watch neural net


def line_series(xs, ys, keys=None, title=None, xname=None):
    data = []
    if not isinstance(xs[0], Sequence):
        xs = [xs for _ in range(len(ys))]
    assert len(xs) == len(ys), "Number of x-lines and y-lines must match"
    for i, series in enumerate([list(zip(xs[i], ys[i])) for i in range(len(xs))]):
        for x, y in series:
            if keys is None:
                key = "key_{}".format(i)
            else:
                key = keys[i]
            data.append([x, key, y])

    table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])

    return wandb.plot_table(
        "wandb/lineseries/v0",
        table,
        {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
        {"title": title, "xname": xname or "x"},
    )


MAX_R = [0]
MIN_C = [1000]

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

        a_pred = clone_pi(torch.tensor(states).float())
        loss = criterion(a_pred, torch.tensor(actions))
        
        # print("Loss!", loss)
        total_loss += loss.item()
        loss.backward()
        if i % 20 == 19:
            print('Epoch:%d Batch:%d Loss:%.3f' % (epoch, i + 1, total_loss / 20))
            total_loss = 0
            epoch_metrics = {'20it average epoch loss': total_loss / 20}
            wandb.log(epoch_metrics)

        pi_optimizer.step()



    if efficiency_eval:
        max_return = 0
        min_cost = 0
        for _ in range(5):
            obs = env.reset()
            done = False
            steps = 0

            while not done:
                a = clone_pi(torch.tensor(obs).float())
                obs, r, done, info = env.step(a.detach().numpy())
                cost = info['cost']

                max_return = max(max_return, r)
                min_cost  = min(min_cost, cost)

                steps += 1
                if steps >= MAX_STEPS:
                    break
        print("max return: ", max_return)
        MAX_R.append(max(max_return, max(MAX_R)))
        MIN_C.append(min(min_cost, min(MIN_C)))
        print("MAX R: ", MAX_R)

        best_metrics = {'max return': max_return, 'min cost': min_cost}
        wandb.log(best_metrics)



    # Save model and save last trajectory
    if (epoch % save_every == 0) or (epoch == epochs - 1):
        logger.save_state({'env': env}, None)

columns = ['Max Return']
# xs = list([[i] for i in range(epochs)])
xs = list([i for i in range(epochs)])
print("xs ", xs)
MAX_R.pop(0)
ys = [MAX_R]
# ys = [[ys[i]] for i in range(len(ys))]

wandb.log({"efficiency gains": line_series(xs=xs, ys=ys, keys=None, title="Learning Efficiency Metrics")})



# wandb.log({"effiency gains": wandb.plot.line(x=xs, y=ys,  title="Learning Efficiency Metrics")})



wandb.finish()

# periodically, during the behavioral cloning, evaluate model against expert policy
# average episodic return
# investigate how much data I need
# train until loss tops out
# what is the best performance for behavioral cloning at the best point


# Play episodes and record
if record_clone:
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
    for i in range(record_tests):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        totalc = 0.
        steps = 0

        while not done:
            # a_mu, a_sigma = clone_pi(torch.from_numpy(obs).float())
            # a = Normal(loc=a_mu, scale=a_sigma).sample()
            a = clone_pi(torch.tensor(obs).float())

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


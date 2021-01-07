
import sys
from cpprb import ReplayBuffer
import numpy as np

print(sys.path)

from torch.optim import Adam
from torch.nn import Parameter

from adabelief_pytorch import AdaBelief

import gym
import safety_gym
from safety_gym.envs.engine import Engine

from spinup_utils import *
from ppo_algos import *
from agent_types import *

import wandb
# wandb.login()
PROJECT_NAME = 'ppo-agent-bayes-small'
wandb.init(project=PROJECT_NAME)

from ppo_function_only import ppo




# Experimentation
# We have two experimentation options: Experiment Grid and wandb hyperparameter sweep
hyperparameter_defaults = dict(
    hid = 64,
    l = 2,
    gamma = 0.99,
    cost_gamma = 0.99,
    seed = 0,
    cost_lim = 10,
    steps = 4000,
    epochs = 50,
    )


sweep_config = {
  "name": "New Sweep",
  # "method": "grid",  # think about switching to bayes
    "method": "bayes",
    "metric": {
        "name": "value",
        "goal": "maximize"
    },
  "parameters": {
        "hid": {
            "values": [64, 128]
        },
        "l": {
            "values" : [1, 2, 4]
        },
        "gamma": {
            # "values": [ 0.98, 0.985, 0.99, 0.995]
            "min": 0.98,
            "max": 0.995
        },
      "cost_gamma": {
          # "values": [ 0.98, 0.985, 0.99, 0.995]
          "min": 0.98,
          "max": 0.995
      },
        "seed": {
            "values" : [0, 99, 999]
        },
        "cost_lim": {
            "min" : 0,
            "max" : 25
        },
        "epochs": {
            "values" : [100, 500, 1000]
        }
    }
}

exp_name = 'exp0'

def safe_ppo_train():
    run = wandb.init(project="safe-ppo-agent", config=hyperparameter_defaults)
    # print("new seed: ", run.config.seed)

    logger_kwargs = setup_logger_kwargs(exp_name, run.config.seed)


    ppo(lambda: gym.make('Safexp-PointGoal1-v0'),
        actor_critic=MLPActorCritic,
        agent=PPOAgent(),
        ac_kwargs=dict(hidden_sizes=[run.config.hid] * run.config.l),
        seed=0,
        steps_per_epoch=4000,
        epochs=run.config.epochs,
        max_ep_len=1000,
        # Discount factors:
        gamma=run.config.gamma,
        lam=0.97,
        cost_gamma=0.99,
        cost_lam=0.97,
        # Policy Learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=25,
        penalty_init=1.,
        penalty_lr=5e-2,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        train_v_iters=80,
        # Policy Learning:
        pi_lr=3e-4,
        train_pi_iters=80,
        # Clipping
        clip_ratio=0.2,
        logger_kwargs=logger_kwargs,
        save_freq=10)



    print("config:", dict(run.config))



sweep_id = wandb.sweep(sweep_config, entity="feloundou", project=PROJECT_NAME)
wandb.agent(sweep_id, function= safe_ppo_train)

wandb.finish()


# # Experiment Grid
# def test_eg():
#     eg = ExperimentGrid()
#     eg.add('test:a', [1, 2, 3], 'ta', True)
#     eg.add('test:b', [1, 2, 3])
#     eg.add('some', [4, 5])
#     eg.add('why', [True, False])
#     eg.add('huh', 5)
#     eg.add('no', 6, in_name=True)
#     return eg.variants()
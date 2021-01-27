from clone_utils import *
from adabelief_pytorch import AdaBelief

import gym

hid_size = 128
n_layers = 2

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)

# Setup policy, optimizer and criterion
hid_size = 128
n_layers = 2

ac_kwargs = dict(hidden_sizes=[hid_size] * n_layers)
clone_pi = GaussianActor(env.observation_space.shape[0], env.action_space.shape[0], activation=nn.LeakyReLU, **ac_kwargs)

# Optimizer and criterion
pi_optimizer = AdaBelief(clone_pi.parameters(), betas=(0.9, 0.999), eps=1e-16)
criterion = nn.MSELoss()

####################################################################################3
# Create clone
marigold_clone = BehavioralClone(config_name='marigold', record_samples=True, clone_epochs=200,
                                 clone_policy=clone_pi, optimizer=pi_optimizer, criterion=criterion,
                                 seed=0, expert_episodes=1000, replay_buffer_size=10000)

# Get expert samples (prerecorded)
marigold_clone.set_replay_buffer(env=env, get_from_file=True)

# Get expert samples (not prerecorded)
# marigold_clone.set_replay_buffer(env=env, get_from_file=False)

# Train the clone
marigold_clone.train_clone(env=env, batch_size=100, train_iters=100, eval_episodes=5, eval_every=5,
                           eval_sample_efficiency=True, print_every=10, save_every=10)

# Run episode simulations
# marigold_clone.run_clone_sim(env, record_clone=True, num_episodes=100, render=False)
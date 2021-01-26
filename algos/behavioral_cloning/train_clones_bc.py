import numpy as np
from six.moves.collections_abc import Sequence
from adabelief_pytorch import AdaBelief

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict, train
import gym
import wandb

from run_policy_sim_ppo import *

from utils import *
from ppo_algos import *


import safety_gym

# Collect trajectories from the expert
def get_expert_trajectories(config_name='test',
                            pull_from_file=False,
                            num_episodes_play=10,
                            rb_size=100,
                            obs_dim=60,
                            act_dim=2,
                            base_path='',
                            demo_dir=''):

    if pull_from_file:
        print(colorize("Pulling saved expert %s trajectories from file over %d episodes" %
                       (config_name, num_episodes_play), 'blue', bold=True))

        f = open(demo_dir + 'sim_data_' + str(num_episodes_play) + '_buffer.pkl', "rb")
        buffer_file = pickle.load(f)
        f.close()

        data = samples_from_cpprb(npsamples=buffer_file)

        # Reconstruct the data, then pass it to replay buffer
        np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)

        # Create environment
        before_add = create_before_add_func(env)

        replay_buffer = ReplayBuffer(size=rb_size,
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
        print(colorize("Generating expert %s trajectories from file over %d episodes" % (config_name, num_episodes_play),
                       'blue', bold=True))
        file_name = 'ppo_penalized_' + config_name + '_128x4'

        # Load trained policy
        _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
                                            'last', False)

        expert_rb = run_policy(env,
                               get_action,
                               0,
                               num_episodes_play,
                               False,
                               record=not pull_from_file,
                               record_name = 'expert_' + file_name + '_' + str(num_episodes_play) + '_runs',
                               record_project='clone_benchmarking_' + config_name,
                               data_path=DATA_PATH,
                               config_name=config_name,
                               max_len_rb=RB_SIZE)

        replay_buffer = expert_rb

    return replay_buffer


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


def train_clones(config_name,
                 epochs,
                 train_iters,
                 eval_iters,
                 eval_freq,
                 efficiency_eval,
                 replay_buffer,
                 clone_pi,
                 expert_pi,
                 pi_optimizer,
                 save_every):
    print(colorize("Training clones of %s config over %s episodes" % (config_name, epochs),
                   'green', bold=True))

    # Train without environment interaction
    wandb.login()
    # PROJECT_NAME = 'clone_benchmarking_' + config_name
    wandb.init(project='clone_benchmarking_' + config_name, name=fname)
    # wandb.watch(clone_pi)  # watch neural net #only do this when not looping

    print_freq = 20

    AVG_R = []
    AVG_C = []

    # Run the expert a few episodes (20) for a benchmark:
    file_name = 'ppo_penalized_' + config_name + '_128x4'

    # Record 25 episodes from expert if we are evaluating for efficiency
    if efficiency_eval:
        expert_rewards, expert_costs = run_policy(env,
                                                  expert_pi,
                                                  0,
                                                  25,
                                                  False,
                                                  record=False,
                                                  record_name='expert_' + file_name,
                                                  record_project='clone_benchmarking_' + config_name,
                                                  data_path=DATA_PATH,
                                                  config_name=config_name,
                                                  max_len_rb=RB_SIZE,
                                                  benchmark=True)

    for epoch in range(epochs):
        total_loss = 0

        for t in range(train_iters):

            # Sample from the replay buffer
            SAMPLE = replay_buffer.sample(BATCH_SIZE)

            # Observe states and chosen actions from expert seems rewards and
            # costs are not relevant here since clone will not receive them
            states = SAMPLE['obs']
            actions = SAMPLE['act']

            pi_optimizer.zero_grad()

            # Policy loss
            a_pred = clone_pi(torch.tensor(states).float())
            loss = criterion(a_pred, torch.tensor(actions))

            # print("Loss!", loss)
            total_loss += loss.item()
            loss.backward()
            if t % print_freq == print_freq-1:
                print(colorize('Epoch:%d Batch:%d Loss:%.4f' % (epoch, t + 1, total_loss / print_freq), 'yellow', bold=True))
                epoch_metrics = {'20it average epoch loss': total_loss / print_freq}
                wandb.log(epoch_metrics)
                total_loss = 0

            pi_optimizer.step()

        if epoch % eval_freq == eval_freq-1:
            if efficiency_eval:  # should we evaluate sample efficiency?

                avg_expert_rewards = np.mean(expert_rewards)
                avg_expert_costs = np.mean(expert_costs)

                return_list, cost_list = [], []

                for _ in range(eval_iters):

                    obs, done, steps, ep_reward, ep_cost = env.reset(), False, 0, 0, 0

                    while not done:
                        a = clone_pi(torch.tensor(obs).float()) # clone step
                        obs, r, done, info = env.step(a.detach().numpy())
                        cost = info['cost']

                        ep_reward += r
                        ep_cost += cost

                        steps += 1
                        if steps >= MAX_STEPS:
                            break

                    return_list.append(ep_reward)
                    cost_list.append(ep_cost)

                AVG_R.append(np.mean(return_list))
                AVG_C.append(np.mean(cost_list))

                best_metrics = {'avg return': np.mean(return_list), 'avg cost': np.mean(cost_list)}
                wandb.log(best_metrics)

        # Save model and save last trajectory
        if (epoch % save_every == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

    # print("average returns should be going up:", AVG_R)
    # print("average costs should be going down:", AVG_C)

    xs = list([i for i in range(0, epochs, eval_freq)])
    # ys = [AVG_R, AVG_C]
    ys_expert_cost = list([1*avg_expert_costs for _ in range(0, epochs, eval_freq)])
    ys_expert_reward = list([1*avg_expert_rewards for _ in range(0, epochs, eval_freq)])
    # ys = [AVG_R]
    # zs = [AVG_C]

    ys_new = [AVG_R, ys_expert_reward]
    zs_new = [AVG_C, ys_expert_cost]

    # wandb.log({"rewards over training": line_series(xs=xs, ys=ys, keys=["Average Returns"], title="Reward of Clones while Training")})
    #
    # wandb.log({"costs over training": line_series(xs=xs, ys=zs, keys=["Average Costs"],
    #                                            title="Cost of Clones while Training")})

    wandb.log({"rewards over training": line_series(xs=xs, ys=ys_new, keys=["Avg Clone Returns", "Avg Expert Returns"],
                                                    title="Reward of Clones while Training")})

    wandb.log({"costs over training": line_series(xs=xs, ys=zs_new, keys=["Avg Clone Costs", "Avg Expert Costs"],
                                                  title="Cost of Clones while Training")})

    wandb.finish()

# Play episodes and record
def run_clone_sim(config_name, env, fname, record_clone, num_episodes, clone_policy, render):
    print(colorize("Running simulations of trained %s clone on %s environment over %d episodes" % (config_name, env, num_episodes),
                   'red', bold=True))

    if record_clone:
        # Logging
        wandb.login()
        wandb.init(project="clone_benchmarking_" + config_name, name= fname)

        rew_mov_avg_10 = []
        cost_mov_avg_10 = []

        returns = []
        costs = []

        cum_ret = 0
        cum_cost = 0

        # Play clone episodes
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            totalr = 0.
            totalc = 0.
            steps = 0

            while not done:
                a = clone_policy(torch.tensor(obs).float())

                obs, r, done, info = env.step(a.detach().numpy())
                cost = info['cost']
                if render:
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
                             # 'cumulative return': cum_ret,
                             # 'cumulative cost': cum_cost,
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


# periodically, during the behavioral cloning, evaluate model against expert policy
# average episodic return
# investigate how much data I need train until loss tops out
# what is the best performance for behavioral cloning at the best point

# ==================================================================================== #

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'
DATA_PATH = '/home/tyna/Documents/openai/research-project/expert_data/'
base_path = '/home/tyna/Documents/openai/research-project/data/'

# make environment
env = gym.make(ENV_NAME)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

# Training parameters
MAX_STEPS = 1000
BATCH_SIZE = 100
RB_SIZE = 10000
EPOCHS = 100
TRAIN_ITERS = 100

###########
NUM_EXPERT_EPISODES = 10
###########

# SETUP
config_name_list = ['cyan', 'lilly', 'hyacinth', 'rose', 'marigold', 'peony', 'lemon', 'violet']
config_name = 'marigold'

DEMO_DIR = os.path.join(DATA_PATH, config_name + '_episodes/')

# Neural Network Architecture
hid_size = 128
n_layers = 2

# Setup policy, optimizer and criterion
# clone_pi = Policy(env.observation_space.shape[0], hid_size, env.action_space.shape[0])
ac_kwargs = dict(hidden_sizes=[hid_size] * n_layers)
clone_pi = GaussianActor(obs_dim[0], env.action_space.shape[0], activation=nn.LeakyReLU, **ac_kwargs)

# Optimizer and criterion
# pi_optimizer = Adam(clone_pi.parameters(), lr=3e-4, weight_decay=0.0001)
pi_optimizer = AdaBelief(clone_pi.parameters(), betas=(0.9, 0.999), eps=1e-16)
criterion = nn.MSELoss()

# Random seed
seed = 0
seed += 10000 * proc_id()
torch.manual_seed(seed)
np.random.seed(seed)


# Special function to avoid certain slowdowns from PyTorch + MPI combo.
setup_pytorch_for_mpi()

# Trained clone data path
fname = config_name + "_clone_" + str(EPOCHS) + 'ep_' + str(TRAIN_ITERS) + 'trn'+ '_' +  str(NUM_EXPERT_EPISODES) + '_expert_runs'
file_name = 'ppo_penalized_' + config_name + '_128x4'

# Load trained policy
_, expert_pi = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),  'last', False)

# Set up logger and save configuration
logger_kwargs = setup_logger_kwargs(fname, seed, DATA_PATH)
logger = EpochLogger(**logger_kwargs)
logger.setup_pytorch_saver(clone_pi)
# logger.save_config(locals()) # configs for now are in my config file, might revisit this later

# Sync params across processes
sync_params(clone_pi)

# ==================================================================================== #


# Get expert trajectories for first time
# replay_buffer = get_expert_trajectories(config_name=config_name, pull_from_file=False, num_episodes_play=NUM_EXPERT_EPISODES,
#                                         rb_size=RB_SIZE, obs_dim=obs_dim, act_dim=act_dim, base_path=base_path,
#                                         demo_dir=DEMO_DIR)


# EPISODE_PARM_LIST = [1000, 500, 250, 100, 50, 25, 10]
# EPOCHS_SETTINGS=  [10, 25, 50, 100]

EPISODE_PARM_LIST = [1000, 500, 250, 100, 50, 25, 10]
EPOCHS_SETTINGS=  [10, 25, 50]

for expert_pm in EPISODE_PARM_LIST:

    # Get prerecorded trajectories
    replay_buffer = get_expert_trajectories(config_name=config_name, pull_from_file=True, num_episodes_play=expert_pm,
                                            rb_size=RB_SIZE, obs_dim=obs_dim, act_dim=act_dim, base_path=base_path,
                                            demo_dir=DEMO_DIR)

    print("got replay buffer for: ", expert_pm)

    for epoch_schedule in EPOCHS_SETTINGS:
        print("training for: ", epoch_schedule)

        train_clones(config_name = config_name, epochs=epoch_schedule, train_iters=30, eval_iters=5, eval_freq=5, efficiency_eval=True,
                     replay_buffer=replay_buffer, clone_pi=clone_pi, expert_pi=expert_pi, pi_optimizer=pi_optimizer, save_every=10)

# run_clone_sim(config_name=config_name, env=env, fname=fname, record_clone=True, num_episodes=100, clone_policy=clone_pi, render=True)

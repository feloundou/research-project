from abc import ABC, abstractmethod
from cpprb import ReplayBuffer, create_before_add_func
import wandb

import pickle
from utils import *
from neural_nets import *

from six.moves.collections_abc import Sequence
from run_policy_sim_ppo import load_policy_and_env, run_policy

# Representation utils
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


# Plot utils

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


# Abstract clone class

class Agent(ABC):
    """
    Abstract clone class
    """

    @abstractmethod
    def set_replay_buffer(self):
        """
        Set replay buffer from expert policies, passively
        fetching or actively recording.
        """

    @abstractmethod
    def train_clone(self):
        """ Train a Clone according to its designated policy/
        Args:
            env: Str. Chosen Gym or Safety Gym environment
            batch_size: Int. Size of sampling batches over the replay buffer.
            train_iters: Int. Number of times to sample the replay buffer.
            eval_episodes: Int. Number of episodes over which to evaluate the current clone policy.
            eval_every: Int. Interval of epochs over which to evaluate the current clone policy.
            eval_sample_efficiency: Bool. Whether or not to evaluate sample efficiency.
            print_every: Int. Frequency at which to print loss output.
            save_every: Int. Frequency at which to save clone policy.
        Returns:
            Trained clone

        """
        pass

    @abstractmethod
    def run_clone_sim(self):
        "Run episodes from a pre-trained clone policy"
        pass


class PPOExpert(Agent):
    """
    Clone class for Sampler.
    """

    def __init__(self,
                 config_name,
                 record_samples,
                 clone_policy,
                 optimizer,
                 criterion,
                 seed,
                 expert_episodes,
                 clone_epochs,
                 replay_buffer_size,
                 # evaluation=False,
                 # store_samples=True
                 ):
        self.config_name = config_name
        self.record_samples = record_samples
        self.expert_episodes = expert_episodes
        self.clone_epochs = clone_epochs
        self.clone_policy = clone_policy
        self.optimizer = optimizer
        self.criterion = criterion
        self.seed = seed
        self.replay_buffer = None
        self.replay_buffer_size = replay_buffer_size
        self.table_name = 'Trained Expert ' + str(self.clone_epochs) + ' Epochs: '


        self.max_steps = 1000

        # Paths
        self._project_dir = '/home/tyna/Documents/openai/research-project/'
        self._root_data_path = self._project_dir + 'data/'
        self._expert_path = self._project_dir + 'expert_data/'
        self._clone_path = self._project_dir + 'clone_data/'
        self._demo_dir = os.path.join(self._expert_path, self.config_name + '_episodes/')
        self._clone_dir = os.path.join(self._expert_path, self.config_name + '_episodes/')
        self.file_name = 'ppo_penalized_' + self.config_name + '_128x4'
        self.benchmark_project_name = 'clone_benchmarking_' + self.config_name

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed # seed = 0
        self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def record_replay_buffer(self, env, num_episodes):

        def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, record=False,
                       record_project='benchmarking', record_name='trained', data_path='', config_name='test',
                       max_len_rb=100, benchmark=False, log_prefix=''):
            assert env is not None, \
                "Environment not found!\n\n It looks like the environment wasn't saved, " + \
                "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
                "page on Experiment Outputs for how to handle this situation."

            logger = EpochLogger()
            o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
            ep_cost = 0
            local_steps_per_epoch = int(4000 / num_procs())

            obs_dim = env.observation_space.shape
            act_dim = env.action_space.shape

            rew_mov_avg_10 = []
            cost_mov_avg_10 = []

            if benchmark:
                ep_costs = []
                ep_rewards = []

            # if record:
            wandb.login()
            # 4 million env interactions
            wandb.init(project=record_project, name=record_name)

            rb = ReplayBuffer(size=10000,
                              env_dict={
                                  "obs": {"shape": obs_dim},
                                  "act": {"shape": act_dim},
                                  "rew": {},
                                  "next_obs": {"shape": obs_dim},
                                  "done": {}})

            while n < num_episodes:
                if render:
                    env.render()
                    time.sleep(1e-3)

                a = get_action(o)
                next_o, r, d, info = env.step(a)

                if record:
                    # buf.store(next_o, a, r, None, info['cost'], None, None, None)
                    done_int = int(d == True)
                    rb.add(obs=o, act=a, rew=r, next_obs=next_o, done=done_int)

                ep_ret += r
                ep_len += 1
                ep_cost += info['cost']

                # Important!
                o = next_o

                if d or (ep_len == max_ep_len):
                    # finish recording and save csv
                    if record:
                        rb.on_episode_end()

                        # make directory if does not exist
                        if not os.path.exists(data_path + config_name + '_episodes'):
                            os.makedirs(data_path + config_name + '_episodes')

                    if len(rew_mov_avg_10) >= 25:
                        rew_mov_avg_10.pop(0)
                        cost_mov_avg_10.pop(0)

                    rew_mov_avg_10.append(ep_ret)
                    cost_mov_avg_10.append(ep_cost)

                    mov_avg_ret = np.mean(rew_mov_avg_10)
                    mov_avg_cost = np.mean(cost_mov_avg_10)

                    expert_metrics = {log_prefix + 'episode return': ep_ret,
                                      log_prefix + 'episode cost': ep_cost,
                                      log_prefix + '25ep mov avg return': mov_avg_ret,
                                      log_prefix + '25ep mov avg cost': mov_avg_cost
                                      }

                    if benchmark:
                        ep_rewards.append(ep_ret)
                        ep_costs.append(ep_cost)

                    wandb.log(expert_metrics)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                    print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
                    o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
                    n += 1

            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.dump_tabular()

            # if record:
            print("saving final buffer")
            bufname_pk = data_path + config_name + '_episodes/sim_data_' + str(int(num_episodes)) + '_buffer.pkl'
            file_pi = open(bufname_pk, 'wb')
            pickle.dump(rb.get_all_transitions(), file_pi)
            wandb.finish()

            return rb

            if benchmark:
                return ep_rewards, ep_costs



    def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, record=False,
                   record_project='benchmarking', record_name='trained', data_path='', config_name='test',
                   max_len_rb=100, benchmark=False, log_prefix=''):
        assert env is not None, \
            "Environment not found!\n\n It looks like the environment wasn't saved, " + \
            "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
            "page on Experiment Outputs for how to handle this situation."

        logger = EpochLogger()
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        ep_cost = 0
        local_steps_per_epoch = int(4000 / num_procs())

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        rew_mov_avg_10 = []
        cost_mov_avg_10 = []

        if benchmark:
            ep_costs = []
            ep_rewards = []

        while n < num_episodes:
            if render:
                env.render()
                time.sleep(1e-3)

            a = get_action(o)
            next_o, r, d, info = env.step(a)

            ep_ret += r
            ep_len += 1
            ep_cost += info['cost']

            # Important!
            o = next_o

            if d or (ep_len == max_ep_len):
                # finish recording and save csv
                if record:
                    rb.on_episode_end()

                    # make directory if does not exist
                    if not os.path.exists(data_path + config_name + '_episodes'):
                        os.makedirs(data_path + config_name + '_episodes')

                if len(rew_mov_avg_10) >= 25:
                    rew_mov_avg_10.pop(0)
                    cost_mov_avg_10.pop(0)

                rew_mov_avg_10.append(ep_ret)
                cost_mov_avg_10.append(ep_cost)

                mov_avg_ret = np.mean(rew_mov_avg_10)
                mov_avg_cost = np.mean(cost_mov_avg_10)

                expert_metrics = {log_prefix + 'episode return': ep_ret,
                                  log_prefix + 'episode cost': ep_cost,
                                  # 'cumulative return': cum_ret,
                                  # 'cumulative cost': cum_cost,
                                  log_prefix + '25ep mov avg return': mov_avg_ret,
                                  log_prefix + '25ep mov avg cost': mov_avg_cost
                                  }

                if benchmark:
                    ep_rewards.append(ep_ret)
                    ep_costs.append(ep_cost)

                wandb.log(expert_metrics)
                logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
                o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
                n += 1

        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()



        if benchmark:
            return ep_rewards, ep_costs


    def train_clone(self, env, batch_size, train_iters, eval_episodes, eval_every, eval_sample_efficiency, print_every, save_every):
        # File Names

        self.fname = self.config_name + "_clone_" + str(self.clone_epochs) + 'ep_' + str(train_iters) + 'trn' + '_' + \
                     str(self.expert_episodes) + '_expert_runs'

        # Load trained policy
        _, expert_pi = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'), 'last', False)

        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(self.fname, self.seed, self._clone_path)
        logger = EpochLogger(**logger_kwargs)
        logger.setup_pytorch_saver(self.clone_policy)

        print(colorize("Training clones of %s config over %s episodes" % (self.config_name, self.clone_epochs),
                       'green', bold=True))

        # Train without environment interaction
        wandb.login()
        wandb.init(project=self.benchmark_project_name, name=self.fname)

        wandb.watch(self.clone_policy)  # watch neural net #only do this when not looping

        AVG_R = []
        AVG_C = []

        tb_name = str(self.clone_epochs) + ' Epochs '

        # Run the expert a few episodes (20) for a benchmark:
        # Record 25 episodes from expert if we are evaluating for efficiency
        if eval_sample_efficiency:
            expert_rewards, expert_costs = run_policy(env,
                                                      expert_pi,
                                                      0,
                                                      25,
                                                      False,
                                                      record=False,
                                                      record_name='expert_' + self.file_name,
                                                      record_project='clone_benchmarking_' + self.config_name,
                                                      data_path= self._expert_path,
                                                      config_name=self.config_name,
                                                      max_len_rb=  self.replay_buffer_size,
                                                      benchmark=True,
                                                      log_prefix=tb_name)

        for epoch in range(self.clone_epochs):
            total_loss = 0

            for t in range(train_iters):

                # Sample from the replay buffer
                SAMPLE = self.replay_buffer.sample(batch_size)

                # Observe states and chosen actions from expert seems rewards and
                # costs are not relevant here since clone will not receive them
                states = SAMPLE['obs']
                actions = SAMPLE['act']

                self.optimizer.zero_grad()

                # Policy loss
                a_pred = self.clone_policy(torch.tensor(states).float())
                loss = self.criterion(a_pred, torch.tensor(actions))

                # print("Loss!", loss)
                total_loss += loss.item()
                loss.backward()
                if t % print_every == print_every - 1:
                    print(
                        colorize('Epoch:%d Batch:%d Loss:%.4f' % (epoch, t + 1, total_loss / print_every), 'yellow',
                                 bold=True))
                    epoch_metrics = {'Avg Epoch Loss': total_loss / print_every}
                    wandb.log(epoch_metrics)
                    total_loss = 0

                self.optimizer.step()

            if epoch % eval_every == eval_every - 1:
                if eval_sample_efficiency: # should we evaluate sample efficiency?

                    avg_expert_rewards = np.mean(expert_rewards)
                    avg_expert_costs = np.mean(expert_costs)

                    return_list, cost_list = [], []

                    for _ in range(eval_episodes):

                        obs, done, steps, ep_reward, ep_cost = env.reset(), False, 0, 0, 0

                        while not done:
                            a = self.clone_policy(torch.tensor(obs).float())  # clone step
                            obs, r, done, info = env.step(a.detach().numpy())
                            cost = info['cost']

                            ep_reward += r
                            ep_cost += cost

                            steps += 1
                            if steps >= self.max_steps:
                                break

                        return_list.append(ep_reward)
                        cost_list.append(ep_cost)

                    AVG_R.append(np.mean(return_list))
                    AVG_C.append(np.mean(cost_list))


                    best_metrics = {tb_name + 'Avg Return': np.mean(return_list), tb_name + 'Avg Cost': np.mean(cost_list)}
                    wandb.log(best_metrics)

            # Save model and save last trajectory
            if (epoch % save_every == 0) or (epoch == self.clone_epochs - 1):
                logger.save_state({'env': env}, None)

        xs = list([i for i in range(0, self.clone_epochs, eval_every)])
        ys_expert_cost = list([1 * avg_expert_costs for _ in range(0, self.clone_epochs, eval_every)])
        ys_expert_reward = list([1 * avg_expert_rewards for _ in range(0, self.clone_epochs, eval_every)])


        ys_new = [AVG_R, ys_expert_reward]
        zs_new = [AVG_C, ys_expert_cost]

        rew_keys = ["Avg Clone Returns", "Avg Expert Returns"]
        cost_keys = ["Avg Clone Costs", "Avg Expert Costs"]

        rew_keys_mod = [ tb_name + sub for sub in rew_keys]
        cost_keys_mod = [tb_name + sub for sub in cost_keys]

        wandb.log({"rewards over training": line_series(xs=xs, ys=ys_new, keys=rew_keys_mod,
                                                        title= tb_name + "Clone Rewards while Training")})
        wandb.log(
            {"costs over training": line_series(xs=xs, ys=zs_new, keys=cost_keys_mod,
                                                title= tb_name + "Clone Costs while Training")})

        wandb.finish()

    def run_expert_sim(self, env, record_clone, num_episodes, render, input_vector=[1,0]):
        print(colorize("Running simulations of trained %s expert on %s environment over %d episodes" % (
        self.config_name, env, num_episodes),
                       'red', bold=True))

        def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, record=False,
                       record_project='benchmarking', record_name='trained', data_path='', config_name='test',
                       max_len_rb=100, benchmark=False, log_prefix=''):

            assert env is not None, \
                "Environment not found!\n\n It looks like the environment wasn't saved, " + \
                "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
                "page on Experiment Outputs for how to handle this situation."

            logger = EpochLogger()
            o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
            ep_cost = 0
            local_steps_per_epoch = int(4000 / num_procs())

            obs_dim = env.observation_space.shape
            act_dim = env.action_space.shape

            rew_mov_avg_10 = []
            cost_mov_avg_10 = []

            if benchmark:
                ep_costs = []
                ep_rewards = []

            if record:
                wandb.login()
                # 4 million env interactions
                wandb.init(project=record_project, name=record_name)

                # buf = CostPOBuffer(obs_dim, act_dim, local_steps_per_epoch, 0.99, 0.99)

                rb = ReplayBuffer(size=10000,
                                  env_dict={
                                      "obs": {"shape": obs_dim},
                                      "act": {"shape": act_dim},
                                      "rew": {},
                                      "next_obs": {"shape": obs_dim},
                                      "done": {}})

                columns = ['observation', 'action', 'reward', 'cost', 'done']
                # sim_data = pd.DataFrame(index=[0], columns=columns)

            while n < num_episodes:
                if render:
                    env.render()
                    time.sleep(1e-3)

                a = get_action(o)
                next_o, r, d, info = env.step(a)

                if record:
                    # buf.store(next_o, a, r, None, info['cost'], None, None, None)
                    done_int = int(d == True)
                    rb.add(obs=o, act=a, rew=r, next_obs=next_o, done=done_int)

                ep_ret += r
                ep_len += 1
                ep_cost += info['cost']

                # Important!
                o = next_o

                if d or (ep_len == max_ep_len):
                    # finish recording and save csv
                    if record:
                        rb.on_episode_end()

                        # make directory if does not exist
                        if not os.path.exists(data_path + config_name + '_episodes'):
                            os.makedirs(data_path + config_name + '_episodes')

                        # buf = CostPOBuffer(obs_dim, act_dim, local_steps_per_epoch, 0.99, 0.99)

                    if len(rew_mov_avg_10) >= 25:
                        rew_mov_avg_10.pop(0)
                        cost_mov_avg_10.pop(0)

                    rew_mov_avg_10.append(ep_ret)
                    cost_mov_avg_10.append(ep_cost)

                    mov_avg_ret = np.mean(rew_mov_avg_10)
                    mov_avg_cost = np.mean(cost_mov_avg_10)

                    expert_metrics = {log_prefix + 'episode return': ep_ret,
                                      log_prefix + 'episode cost': ep_cost,
                                      # 'cumulative return': cum_ret,
                                      # 'cumulative cost': cum_cost,
                                      log_prefix + '25ep mov avg return': mov_avg_ret,
                                      log_prefix + '25ep mov avg cost': mov_avg_cost
                                      }

                    if benchmark:
                        ep_rewards.append(ep_ret)
                        ep_costs.append(ep_cost)

                    wandb.log(expert_metrics)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                    print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
                    o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
                    n += 1

            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.dump_tabular()

            if record:
                print("saving final buffer")
                bufname_pk = data_path + config_name + '_episodes/sim_data_' + str(int(num_episodes)) + '_buffer.pkl'
                file_pi = open(bufname_pk, 'wb')
                pickle.dump(rb.get_all_transitions(), file_pi)
                wandb.finish()

                return rb

            if benchmark:
                return ep_rewards, ep_costs


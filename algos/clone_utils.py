from abc import ABC, abstractmethod
from cpprb import ReplayBuffer, create_before_add_func
import wandb

import pickle
from utils import *
from ppo_algos import *

from six.moves.collections_abc import Sequence


from run_policy_sim_ppo import load_policy_and_env, run_policy


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

class Clone(ABC):
    """
    Abstract clone class
    """

    # @abstractmethod
    # def act(self):
    #     """
    #     Select an action for evaluation.
    #     If the agent has a replay-buffer, state and reward are stored.
    #     Args:
    #         state (rlil.environment.State): The environment state at the current timestep.
    #         reward (torch.Tensor): The reward from the previous timestep.
    #     Returns:
    #         rllib.Action: The action to take at the current timestep.
    #     """
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




class BehavioralClone(Clone):
    """
    Agent class for Sampler.
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


        self.max_steps = 1000

        # Paths
        self._project_dir = '/home/tyna/Documents/openai/research-project/'
        self._root_data_path = self._project_dir + 'data/'
        self._expert_path = self._project_dir + 'expert_data/'
        self._demo_dir = os.path.join(self._expert_path, self.config_name + '_episodes/')
        self.file_name = 'ppo_penalized_' + self.config_name + '_128x4'

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed # seed = 0
        self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # for N step replay buffer
        # self._n_step, self._discount_factor = get_n_step()
        # if self._evaluation:
        #     self._n_step = 1  # disable Nstep buffer when evaluation mode

    def set_replay_buffer(self, env, get_from_file):

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        if get_from_file:
            print(colorize("Pulling saved expert %s trajectories from file over %d episodes" %
                           (self.config_name, self.expert_episodes), 'blue', bold=True))

            f = open(self._demo_dir + 'sim_data_' + str(self.expert_episodes) + '_buffer.pkl', "rb")
            buffer_file = pickle.load(f)
            f.close()

            data = samples_from_cpprb(npsamples=buffer_file)

            # Reconstruct the data, then pass it to replay buffer
            np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)

            # Create environment
            before_add = create_before_add_func(env)

            replay_buffer = ReplayBuffer(size= self.replay_buffer_size,
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
            self.replay_buffer = replay_buffer

        else:
            # Generate expert data
            print(colorize(
                "Generating expert %s trajectories from file over %d episodes" % (self.config_name, self.expert_episodes),
                'blue', bold=True))

            # Load trained policy
            _, get_action = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'),
                                                'last', False)
            expert_rb = run_policy(env,
                                   get_action,
                                   0,
                                   self.expert_episodes,
                                   False,
                                   record=not get_from_file,
                                   record_name='expert_' + self.file_name + '_' + str(self.expert_episodes) + '_runs',
                                   record_project='clone_benchmarking_' + self.config_name,
                                   data_path= self._expert_path,
                                   config_name= self.config_name,
                                   max_len_rb=self.replay_buffer_size)

            self.replay_buffer = expert_rb


    def train_clone(self, env, batch_size, train_iters, eval_episodes, eval_every, eval_sample_efficiency, print_every, save_every):
        # File Names
        self.fname = self.config_name + "_clone_" + str(self.clone_epochs) + 'ep_' + str(train_iters) + 'trn' + '_' + \
                     str(self.expert_episodes) + '_expert_runs'

        # Load trained policy
        _, expert_pi = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'), 'last', False)

        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(self.fname, self.seed, self._expert_path)
        logger = EpochLogger(**logger_kwargs)
        logger.setup_pytorch_saver(self.clone_policy)

        print(colorize("Training clones of %s config over %s episodes" % (self.config_name, self.clone_epochs),
                       'green', bold=True))

        # Train without environment interaction
        wandb.login()
        wandb.init(project='clone_benchmarking_' + self.config_name, name=self.fname)

        wandb.watch(self.clone_policy)  # watch neural net #only do this when not looping

        AVG_R = []
        AVG_C = []
        # tb_name = str(self.clone_epochs) + ' Epochs and ' + str(train_iters) + ' '

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
        # ys = [AVG_R, AVG_C]
        ys_expert_cost = list([1 * avg_expert_costs for _ in range(0, self.clone_epochs, eval_every)])
        ys_expert_reward = list([1 * avg_expert_rewards for _ in range(0, self.clone_epochs, eval_every)])
        # ys = [AVG_R]
        # zs = [AVG_C]

        ys_new = [AVG_R, ys_expert_reward]
        zs_new = [AVG_C, ys_expert_cost]

        rew_keys = ["Avg Clone Returns", "Avg Expert Returns"]
        cost_keys = ["Avg Clone Costs", "Avg Expert Costs"]

        rew_keys_mod = [ tb_name + sub for sub in rew_keys]
        cost_keys_mod = [tb_name + sub for sub in cost_keys]

        wandb.log({"rewards over training": line_series(xs=xs, ys=ys_new, keys=rew_keys_mod,
                                                        title= tb_name + "Reward of Clones while Training")})
        wandb.log(
            {"costs over training": line_series(xs=xs, ys=zs_new, keys=cost_keys_mod,
                                                title= tb_name + "Cost of Clones while Training")})

        # wandb.log({"rewards over training": line_series(xs=xs, ys=ys_new,
        #                                                 keys=["Avg Clone Returns", "Avg Expert Returns"],
        #                                                 title="Reward of Clones while Training")})
        # wandb.log(
        #     {"costs over training": line_series(xs=xs, ys=zs_new, keys=["Avg Clone Costs", "Avg Expert Costs"],
        #                                         title="Cost of Clones while Training")})

        wandb.finish()

    # def act(self, states, reward):
    #     """
    #     In the act function, the lazy_agent put a sample
    #     (last_state, last_action, reward, states) into self.replay_buffer.
    #     Then, it outputs a corresponding action.
    #     """
    #     if self._store_samples:
    #         assert self.replay_buffer is not None, \
    #             "Call self.set_replay_buffer(env) at lazy_agent initialization."
    #         samples = Samples(self._states, self._actions, reward, states)
    #         self.replay_buffer.store(samples)

    def run_clone_sim(self, env, record_clone, num_episodes, render):
        print(colorize("Running simulations of trained %s clone on %s environment over %d episodes" % (
        self.config_name, env, num_episodes),
                       'red', bold=True))

        if record_clone:
            # Logging
            wandb.login()
            wandb.init(project="clone_benchmarking_" + self.config_name, name=self.fname)

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
                    a = self.clone_policy(torch.tensor(obs).float())

                    obs, r, done, info = env.step(a.detach().numpy())
                    cost = info['cost']
                    if render:
                        env.render()
                    totalr += r
                    totalc += cost
                    steps += 1
                    if steps % 100 == 0: print("%i/%i" % (steps, self.max_steps))
                    if steps >= self.max_steps:
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
                mov_avg_cost = np.mean(cost_mov_avg_10)

                table_name = 'Trained Clone ' + str(self.clone_epochs) + ' Epochs:'

                clone_metrics = {table_name + 'episode return': totalr,
                                 table_name + 'episode cost': totalc,
                                 table_name + '25ep mov avg return': mov_avg_ret,
                                 table_name + '25ep mov avg cost': mov_avg_cost
                                 }
                wandb.log(clone_metrics)

            wandb.finish()

            print('Returns', returns)
            print('Avg EpRet', np.mean(returns))
            print('Std EpRet', np.std(returns))
            print('Costs', costs)
            print('Avg EpCost', np.mean(costs))
            print('Std EpCost', np.std(costs))


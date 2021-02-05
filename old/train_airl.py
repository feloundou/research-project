from torch.optim import Adam
from adabelief_pytorch import AdaBelief
import math

import gym

from agent_types import *
import pickle

import wandb

from utils import *



# Define PPO functions
def airl(env_fn,
        actor_critic=MLPActorCritic,
        discrim = Discriminator,
        agent=PPOAgent(),
        ac_kwargs=dict(),
        seed=0,
        # Experience Collection
        steps_per_epoch=4000,
        epochs=50,
        max_ep_len=1000,
        # Discount factors:
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99,
        cost_lam=0.97,
        # Policy Learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=25,
        penalty_init=1.,
        penalty_lr=5e-3,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        train_v_iters=100,
        # Policy Learning:
        pi_lr=3e-4,
        train_pi_iters=100,
        # Discriminator Learning:
        discrim_lr= 1e-3,
        train_discrim_iters=100,
        # Clipping
        clip_ratio=0.2,
        logger_kwargs=dict(),
        # Experimenting
        config_name = 'standard',
        save_every=10):
    """
    """

    # def fc_reward(env, hidden1=400, hidden2=300):
    #     return nn.Sequential(
    #         nn.Linear(env.state_space.shape[0] +
    #                   env.action_space.shape[0], hidden1),
    #         nn.LeakyReLU(),
    #         nn.Linear(hidden1, hidden2),
    #         nn.LeakyReLU(),
    #         nn.Linear(hidden2, 1)
    #     )

    # Instantiate environment
    env = env_fn()

    # env = GymEnvironment('Safexp-PointGoal1-v0', append_time=True)
    replay_buffer = ExperienceReplayBuffer(1000, env)

    # base buffer
    states = State(torch.tensor([env.observation_space.sample()] * 100))
    actions = Action(torch.tensor([env.action_space.sample()] * 99))
    rewards = torch.arange(0, 99, dtype=torch.float)
    samples = Samples(states[:-1], actions, rewards, states[1:])
    replay_buffer.store(samples)

    # expert buffer
    exp_replay_buffer = ExperienceReplayBuffer(1000, env)
    exp_states = State(torch.tensor([env.observation_space.sample()] * 100))
    exp_actions = Action(torch.tensor([env.action_space.sample()] * 99))
    exp_rewards = torch.arange(100, 199, dtype=torch.float)
    exp_samples = Samples(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])
    exp_replay_buffer.store(exp_samples)

    # discriminator
    reward_model = fc_reward(env)
    reward_optimizer = Adam(reward_model.parameters())
    reward_fn = Approximation(reward_model, reward_optimizer)

    value_model = fc_v(env)
    value_optimizer = Adam(value_model.parameters())
    value_fn = VNetwork(value_model, value_optimizer)

    # policy
    feature_model, _, policy_model = fc_actor_critic(env)
    feature_optimizer = Adam(feature_model.parameters())
    feature_nw = FeatureNetwork(feature_model, feature_optimizer)


    policy_optimizer = Adam(policy_model.parameters())
    policy = GaussianPolicy(policy_model, policy_optimizer, env.action_space)

    # airl_buffer = AirlWrapper(replay_buffer,
    #                           exp_replay_buffer,
    #                           reward_fn,
    #                           value_fn,
    #                           policy,
    #                           feature_nw=feature_nw)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    # yield airl_buffer, samples



    # objects
    base_agent = base_agent
    replay_buffer = get_replay_buffer()
    reward_fn = replay_buffer.reward_fn
    value_fn = replay_buffer.value_fn
    # writer = get_writer()
    # device = get_device()
    discrim_criterion = nn.BCELoss()
        # hyperparameters
    minibatch_size = minibatch_size
    replay_start_size = replay_start_size
    update_frequency = save_every
    _train_count = 0

    def train(self):
        # train discriminator
        samples, expert_samples = replay_buffer.sample_both(minibatch_size)
        states, actions, _, next_states, _, _ = samples
        exp_states, exp_actions, _, exp_next_states, _, _ = expert_samples

        fake = replay_buffer.discrim(states, actions, next_states)
        real = replay_buffer.discrim(exp_states,
                                          exp_actions,
                                          exp_next_states)
        discrim_loss = discrim_criterion(fake, torch.ones_like(fake)) + \
                       discrim_criterion(real, torch.zeros_like(real))

        reward_fn.zero_grad()
        value_fn.zero_grad()
        discrim_loss.backward()
        reward_fn.reinforce()
        value_fn.reinforce()

            # additional debugging info
            # self.writer.add_scalar('airl/fake', fake.mean())
            # self.writer.add_scalar('airl/real', real.mean())


    # W&B Logging
    wandb.login()

    config_name = 'marigold'

    composite_name = 'ppo_penalized_' + config_name + '_' + str(int(steps_per_epoch/1000)) + \
                     'Ks_' + str(epochs) + 'e_' + str(ac_kwargs['hidden_sizes'][0]) + 'x' + \
                     str(len(ac_kwargs['hidden_sizes']))

    # 4 million env interactions
    wandb.init(project="vail-experts-1000epochs", group="airl_runs", name='vail_'+composite_name)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)




    # Paths
    _project_dir = '/home/tyna/Documents/openai/research-project/'
    _root_data_path = _project_dir + 'data/'
    _expert_path = _project_dir + 'expert_data/'
    _clone_path = _project_dir + 'clone_data/'
    _demo_dir = os.path.join(_expert_path, config_name + '_episodes/')

    # load demonstrations
    # expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
    # demonstrations = np.array(expert_demo)
    # print("demonstrations.shape", demonstrations.shape)

    f = open(_demo_dir + 'sim_data_' + str(1000) + '_buffer.pkl', "rb")
    buffer_file = pickle.load(f)
    f.close()

    expert_demonstrations = samples_from_cpprb(npsamples=buffer_file)

    # Reconstruct the data, then pass it to replay buffer
    np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(expert_demonstrations)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    running_state = ZFilter((obs_dim[0],), clip=1)


    # Create actor-critic module and monitor it
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    discrim = discrim(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)
    # Note, also sync for Discriminator
    sync_params(discrim)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v, discrim])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t discrim: %d \n' % var_counts)

    z_filter = False

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CostPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, cost_gamma, cost_lam)

    pi_optimizer = AdaBelief(ac.pi.parameters(), betas=(0.9, 0.999), eps=1e-8)
    vf_optimizer = AdaBelief(ac.v.parameters(), betas=(0.9, 0.999), eps=1e-8)
    discrim_optimizer = AdaBelief(discrim.parameters(), betas=(0.9, 0.999), eps=1e-8)

    penalty = np.log(max(np.exp(penalty_init)-1, 1e-8))

    mov_avg_ret = 0
    mov_avg_cost = 0

    # Discriminator reward
    def get_reward(discrim, state, action):
        state = torch.Tensor(state)
        action = torch.Tensor(action)
        state_action = torch.cat([state, action])
        with torch.no_grad():
            return -math.log(discrim(state_action)[0].item())

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up functions for computing value loss(es)
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        v_loss = ((ac.v(obs) - ret) ** 2).mean()
        return v_loss

    def compute_loss_discrim(data, demonstrations, acc=False):
        obs = data['obs']
        act = data['act']

        criterion = torch.nn.BCELoss()

        # change demo format
        demonstrations = torch.Tensor(demonstrations)

        # Pass both expert and learner through discriminator
        learner = discrim(torch.cat([obs, act], dim=1))
        expert = discrim(demonstrations)

        learner_acc = (learner  > 0.5).float().mean()
        expert_acc = (expert < 0.5).float().mean()

        discrim_loss = criterion(learner, torch.ones((obs.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        if acc:
            return discrim_loss, expert_acc, learner_acc
        else:
            return discrim_loss


    # Set up model saving
    logger.setup_pytorch_saver(ac)

    penalty_init_param = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    TRAIN_DISC = True

    def update(cur_penalty, TRAIN_DISC):

        cur_cost = logger.get_stats('EpCost')[0]
        cur_rew = logger.get_stats('EpRet')[0]

        if len(rew_mov_avg_10) >= 10:
            rew_mov_avg_10.pop(0)
            cost_mov_avg_10.pop(0)

        rew_mov_avg_10.append(cur_rew)
        cost_mov_avg_10.append(cur_cost)

        mov_avg_ret  = np.mean(rew_mov_avg_10)
        mov_avg_cost = np.mean(cost_mov_avg_10)

        c = cur_cost - cost_lim

        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        # c is the safety constraint
        print("current cost: ", cur_cost)

        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)

        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        combined_expert_demos = np.concatenate((np_states, np_actions), axis=1)
        discrim_l_old = compute_loss_discrim(data, combined_expert_demos, acc=False).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Discriminator learning
        if TRAIN_DISC:
            for i in range(train_discrim_iters):
                discrim_optimizer.zero_grad()
                loss_discrim, expert_acc, learner_acc = compute_loss_discrim(data, combined_expert_demos, acc=True)
                print("discriminator loss: ", loss_discrim)
                loss_discrim.backward()
                mpi_avg_grads(discrim)  # average grads across MPI processes
                discrim_optimizer.step()

            if expert_acc.item() > 0.99 and learner_acc.item() > 0.98:

                TRAIN_DISC = False


        # Penalty update
        print("old penalty: ", cur_penalty)
        cur_penalty = max(0, cur_penalty + penalty_lr*(cur_cost - cost_lim))
        print("new penalty: ", cur_penalty)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     LossDiscrim=discrim_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     # DeltaLossDiscrim=(loss_discrim.item() - discrim_l_old)
                     )

        vf_loss_avg = mpi_avg(v_l_old)
        pi_loss_avg = mpi_avg(pi_l_old)

        update_metrics = {'10p mov avg ret': mov_avg_ret,
                          '10p mov avg cost': mov_avg_cost,
                          'value function loss': vf_loss_avg,
                          'policy loss': pi_loss_avg,
                          'current penalty': cur_penalty
                          }

        wandb.log(update_metrics)
        # return cur_penalty, train_discriminator
        return cur_penalty, TRAIN_DISC

    # Prepare for interaction with environment
    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len, cum_cost, cum_reward = env.reset(), 0, False, 0, 0, 0, 0, 0, 0


    rew_mov_avg_10 = []
    cost_mov_avg_10 = []

    cur_penalty = penalty_init_param

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        for t in range(local_steps_per_epoch):
            state = running_state(o)

            if z_filter:
                a, v, vc, logp = ac.step(torch.as_tensor(state, dtype=torch.float32))
            else:
                a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))


            # env.step => Take action
            next_o, r, d, info = env.step(a)

            if z_filter:
                next_o = running_state(next_o)

            irl_reward = get_reward(discrim, o, a)

            # Include penalty on cost
            c = info.get('cost', 0)

            # Track cumulative cost over training
            cum_reward += r
            cum_cost += c

            ep_ret += r
            ep_cost += c
            ep_len += 1

            r_total = r - cur_penalty * c
            r_total /= (1 + cur_penalty)

            irl_updated = irl_reward - cur_penalty*c
            irl_updated /= (1 + cur_penalty)


            buf.store(o, a, irl_updated, v, 0, 0, logp, info)

            # save and log
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))

                    last_v = v
                    last_vc = 0

                else:
                    last_v = 0
                buf.finish_path(last_v, last_vc)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    print("end of episode return: ", ep_ret)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)

                    # average ep ret and cost
                    avg_ep_ret = ep_ret
                    avg_ep_cost = ep_cost

                    episode_metrics = {'average ep ret': avg_ep_ret, 'average ep cost': avg_ep_cost}

                    wandb.log(episode_metrics)

                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

        # Save model and save last trajectory
        if (epoch % save_every == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        cur_penalty, TRAIN_DISC = update(cur_penalty, TRAIN_DISC)

        #  Cumulative cost calculations
        cumulative_cost = mpi_sum(cum_cost)
        cumulative_reward = mpi_sum(cum_reward)

        cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)
        reward_rate = cumulative_reward / ((epoch + 1) * steps_per_epoch)

        log_metrics = {'cost rate': cost_rate, 'reward rate': reward_rate}

        wandb.log(log_metrics)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('LossDiscrim', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('DeltaLossDiscrim', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


def main(config):
    import argparse
    from utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--agent', type=str, default='ppo-lagrange')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cost_lim', type=float, default=25)
    # parser.add_argument('--penalty_lr', type=float, default=0.04)
    parser.add_argument('--config_name', type=str, default='standard')
    parser.add_argument('--penalty_lr', type=float, default=0.005)
    parser.add_argument('--exp_name', type=str, default='ppo_safe')

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    # PROJECT_NAME = args.name
    composite_name = 'ppo_penalized_' + config['name'] + '_' + str(int(args.steps / 1000)) + 'Ks_' + str(
        args.epochs) + 'e_' + str(args.hid) + 'x' + str(args.l)

    print("composite name 2")
    print(composite_name)

    logger_kwargs = setup_logger_kwargs(composite_name, args.seed)

    # Run experiment
    airl(lambda: gym.make(args.env),
        actor_critic=MLPActorCritic,
        agent=PPOAgent(),
        ac_kwargs=dict(hidden_sizes=[config['hid']] * config['l']),
        gamma=config['gamma'],
        lam=config['lam'],
        cost_gamma=args.cost_gamma,
        seed=config['seed'],
        steps_per_epoch=config['steps'],
        epochs=args.epochs,
        cost_lim= config['cost_lim'],
        penalty_lr=config['penalty_lr'],
        config_name= config['name'],
        logger_kwargs=logger_kwargs)

    wandb.config.update(args)
    wandb.finish()

if __name__ == '__main__':

    exec(open('nn_config.py').read())

    main(standard_config)


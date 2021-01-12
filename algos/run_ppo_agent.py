
import sys
from cpprb import ReplayBuffer
import numpy as np

print(sys.path)

from torch.optim import Adam
from adabelief_pytorch import AdaBelief

from torch.nn import Parameter

import gym
import safety_gym
from safety_gym.envs.engine import Engine

from spinup_utils import *
from ppo_algos import *
from agent_types import *

import wandb
wandb.login()

PROJECT_NAME = 'ppo_penalized_500e'
# 4 million env interactions
wandb.init(project="ppo-5000e-expts", name= PROJECT_NAME)

# Define PPO functions
def ppo(env_fn,
        actor_critic=MLPActorCritic,
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
        penalty_lr=5e-2,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        train_v_iters=100,
        # Policy Learning:
        pi_lr=3e-4,
        train_pi_iters=100,
        # Clipping
        clip_ratio=0.2,
        logger_kwargs=dict(),
        # Experimenting
        modify_env = True,
        constrain_mode = True,
        hazards_cost = 1,
        reward_goal = 1,
        save_every=10):
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_every (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    # env.hazards_cost = 3
    # if modify_env == True:
    #     env.hazards_cost = hazards_cost
    #     env.reward_goal = reward_goal
    #
    #     if constrain_mode == False:
    #         env.constrain_hazards  =  False


    print("constraints in the environment")
    print("constrain hazards: ", env.constrain_hazards)
    print("hazards cost: ", env.hazards_cost)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # print("actor critic parameters: ", [p for p in ac.parameters() if p.requires_grad])
    # print("even more info: ", ac.pen.penalty_param )

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # print("Agent parameters")
    # print("Learn penalty:", agent.learn_penalty)
    # print("Use penalty:", agent.use_penalty)
    # print("Objective penalized:", agent.objective_penalized)
    # print("Reward penalized:", agent.reward_penalized)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    buf = CostPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # # Penalty
    # if agent.learn_penalty:
    #     if agent.penalty_param_loss:
    #         penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)
    #     else:
    #         penalty_loss = -penalty * (cur_cost_ph - cost_lim)
    #     train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    penalty = np.log(max(np.exp(penalty_init)-1, 1e-8))

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
        # print("value tensor: ", ac.v(obs))
        return v_loss

    def compute_loss_vc(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        vc_loss = ((ac.vc(obs) - cret) ** 2).mean()
        return vc_loss


    # Set up model saving
    logger.setup_pytorch_saver(ac)

    penalty_init_param = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    def update(cur_penalty):
        # print("Starting update")

        cur_cost = logger.get_stats('EpCost')[0]
        c = cur_cost - cost_lim

        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        # c is the safety constraint

        print("current cost: ", cur_cost)
        # print("cost-limit: ", c)
        # print("penalty: ", pen)

        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        vc_l_old = compute_loss_vc(data).item()


        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            # if kl > 1.5 * target_kl:
            #     logger.log('Early stopping at step %d due to reaching max kl.' % i)
            #     break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            # print("value loss: ", loss_v)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()


        # Penalty update

        cur_penalty = max(0, cur_penalty + penalty_lr*(cur_cost - cost_lim))
        print("new penalty: ", cur_penalty)
        print("current cost: ", cur_cost)
        print("cost limit: ", cost_lim)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

        vf_loss_avg = mpi_avg(v_l_old)
        pi_loss_avg = mpi_avg(pi_l_old)

        update_metrics = {
                        'value function loss': vf_loss_avg,
                          # 'loss vc': vc_l_old,
                          'policy loss': pi_loss_avg,
                          # 'KL': kl
                          'current penalty': cur_penalty
                          }

        wandb.log(update_metrics)

    # Prepare for interaction with environment
    start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0
    o, r, d, c, ep_ret, ep_cost, ep_len, cum_cost, cum_reward = env.reset(), 0, False, 0, 0, 0, 0, 0, 0


    cur_penalty = penalty_init_param


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # env.step => Take action
            next_o, r, d, info = env.step(a)
            # print("outputs: next_o, r, d, info: ", next_o, r, d, info)

            # Include penalty on cost
            c = info.get('cost', 0)

            # Track cumulative cost over training
            cum_reward += r
            cum_cost += c

            ep_ret += r
            ep_cost += c
            ep_len += 1

            # if agent.reward_penalized:
            # print("agent is being penalized")
            r_total = r - cur_penalty * c
            r_total /= (1 + cur_penalty)

            #
            # print("naked rewards:", r)
            # print("penalized rewards:", r_total)


            # print("reward total: ", r_total)
            # buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
            buf.store(o, a, r_total, v, 0, 0, logp, info)
            # else:
            #     buf.store(o, a, r, v, c, vc, logp, info)

            # save and log
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ , _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    last_v = v
                    last_vc = 0
                    # if agent.reward_penalized:
                    #     last_vc = 0
                    # else:
                    #     last_vc = vc
                else:
                    last_v = 0
                buf.finish_path(last_v, last_vc)

                # sess.run looks like ac.step

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    # print("terminal ep ret: ", ep_ret)
                    print("end of episode return: ", ep_ret)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)

                # average ep ret and cost
                avg_ep_ret = ep_ret/ep_len
                avg_ep_cost = ep_cost/ep_len

                episode_metrics = {
                    # 'epoch': epoch,
                    'average ep ret': avg_ep_ret,
                    'average ep cost' : avg_ep_cost
                }

                wandb.log(episode_metrics)

                # o, ep_ret, ep_len, ep_cost = env.reset(), 0, 0, 0
                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

        # Save model and save last trajectory
        if (epoch % save_every == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update(cur_penalty)

        #  Cumulative cost calculations
        cumulative_cost = mpi_sum(cum_cost)
        cumulative_reward = mpi_sum(cum_reward)


        cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)
        reward_rate = cumulative_reward / ((epoch + 1) * steps_per_epoch)

        log_metrics = {
                       # 'epoch': epoch,
                       'value': v,
                       'cost rate': cost_rate,
                       'reward rate' : reward_rate
                       # 'cumulative cost': cumulative_cost,
                       # 'cumulative reward': cumulative_reward,
                       }

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
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    from spinup_utils import setup_logger_kwargs

    # print("Testing experimental grid")
    # print(test_eg())

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--agent', type=str, default='ppo-lagrange')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.98)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cost_lim', type=float, default=25)
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--hazards_cost', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='ppo_safe')
    args = parser.parse_args()

    # print("cpu count", os.cpu_count())
    mpi_fork(args.cpu)  # run parallel code with mpi

    # PROJECT_NAME = args.name
    logger_kwargs = setup_logger_kwargs(PROJECT_NAME, args.seed)

    # Prepare agent
    trpo_kwargs = dict(
        reward_penalized=False, objective_penalized=False,
        learn_penalty=False, penalty_param_loss=False  # Irrelevant in unconstrained
    )

    ppo_lagrange_kwargs = dict(
        reward_penalized=False, objective_penalized=True,
        learn_penalty=True, penalty_param_loss=True  # Irrelevant in unconstrained
    )

    cpo_kwargs = dict(
        reward_penalized=False,  # Irrelevant in CPO
        objective_penalized=False,  # Irrelevant in CPO
        learn_penalty=False,  # Irrelevant in CPO
        penalty_param_loss=False  # Irrelevant in CPO
    )

    # Create agent

    if args.agent == 'ppo-lagrange':
        agent = PPOAgent(**ppo_lagrange_kwargs)
    elif args.agent == 'trpo':
        agent = TRPOAgent(**trpo_kwargs)
    elif args.agent == 'cpo':
        agent = CPOAgent(**cpo_kwargs)

    # Run experiment
    # print("done with PPO, starting with CPO")

    ppo(lambda: gym.make(args.env),
        actor_critic=MLPActorCritic,
        agent=agent,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        cost_gamma = args.cost_gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        cost_lim= args.cost_lim,
        logger_kwargs=logger_kwargs)

# all_args = parser.parse_args()
    wandb.config.update(args)
    wandb.finish()

# reward function: r - penalty*cost

# look at ppo-clip
# update cur_penalty once policy updates.
# initial penalty scalar function
# in the for loop, implement the penalties as either the advantage estimations
# 2 options
# trade-off rewards and penalized cost

# 1. treat penalty coefficient: r - penalty*cost
# 2. in addition to the value function for base reward, additional value function for cost

# penalty coefficient should be updated by a small amt towards the difference between penalty_loss
# update to the lambda coefficient (penalty)

# advantage function for the reward

# keep cost limits below 60-80
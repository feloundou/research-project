import time
import joblib
import os
import os.path as osp
import torch
from spinup_utils import *
import gym
import safety_gym
import pandas as pd
from random import randint
# from safety_gym.envs.engine import Engine


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save along with RL env.
    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.
    loads as if there's a PyTorch save.
    """

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:

        print(osp.join(fpath, 'vars' + itr + '.pkl'))
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    def get_action_cpo(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, record=False, data_path=''):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    ep_cost = 0
    if record:
        columns = ['observation', 'action', 'reward', 'cost', 'done']
        sim_data = pd.DataFrame(index=[0], columns=columns)

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, info = env.step(a)
        if record:
            iteration = dict(observation= [o], action=[a], reward=r, cost = info['cost'], done=d)

            df = pd.DataFrame.from_dict(iteration)
            sim_data = sim_data.append(df)


        # print("obs")
        # print("info:", info['cost'])
        ep_ret += r
        ep_len += 1
        ep_cost += info['cost']

        if d or (ep_len == max_ep_len):
            # finish recording and save csv
            if record:
                # drop first row (NAs) then save and reset
                sim_data = sim_data[1:]
                sname = data_path + 'cyan_episodes/sim_data_' +  str(randint(0, 100000)) + '.csv'
                sname_pk = data_path + 'cyan_episodes/sim_data_' + str(randint(0, 100000)) + '.pkl'
                # sim_data.to_csv(sname, index=False)
                sim_data.to_pickle(sname_pk)
                print(sname)
                sim_data = pd.DataFrame(index=[0], columns=columns)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
            o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,
                        default= '/home/tyna/Documents/openai/research-project/data/ppo_test/ppo_test_s0/')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    # parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    # the safe trained file is ppo_500e_8hz_cost5_rew1_lim25

    # file_name = 'ppo_500e_8hz_cost5_rew1_lim25'
    # file_name = 'ppo_penalized_test'  # second best
    # file_name = 'ppo_penalized_cyan_500ep_8000steps'   # best so far
    # file_name = 'cpo_500e_8hz_cost1_rew1_lim25'  # unconstrained

    # file_name = 'ppo_penalized_500e'
    # file_name = 'ppo_penalized_scarlet_500ep_8000steps_v2'
    # file_name = 'ppo_5000e_8hz_cost1_rew1_lim25'
    # file_name = 'ppo_penalized_lemon_500ep_8000steps'


    base_path = '/home/tyna/Documents/openai/research-project/data/'


    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
    # '/home/tyna/Documents/openai/research-project/data/ppo_500e_8hz_cost1_rew1_lim25/ppo_500e_8hz_cost1_rew1_lim25_s0/',
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)

    env = gym.make('Safexp-PointGoal1-v0')
    run_policy(env, get_action, args.len, args.episodes, not (args.norender), record=False, data_path=base_path)

    # run_policy(env, get_action, args.len, args.episodes, False, record=True, data_path=base_path)




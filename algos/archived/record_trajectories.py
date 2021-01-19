import gym

# import ray

from il_utils import *
from agent_functions import *
# from rlil.utils.writer import DummyWriter

# from rlil.memory import ExperienceReplayBuffer
# from rlil.initializer import set_replay_buffer, get_replay_buffer
# from rlil.samplers import AsyncSampler
# from rlil.environments import GymEnvironment
# from rlil.presets import continuous




def main():
    parser = argparse.ArgumentParser(description="Record a trajectory of trained agent. \
        The trajectory will be stored as transitions.pkl in the args.dir.")
    parser.add_argument("--dir", default= "/home/tyna/Documents/openai/research-project/data/" , help="Directory where the agent's model is saved.")
    parser.add_argument("--device", default="cpu", help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument("--train", action="store_true", help="The model of lazy_agent: evaluation or training.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for training")
    parser.add_argument("--frames", type=int, default=1e6, help="Number of frames to store")
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')

    args = parser.parse_args()
    print("here 1")
    # ray.init(include_webui=False, ignore_reinit_error=True)


    # load env
    if args.dir[-1] != "/":
        args.dir += "/"
    env_id = args.dir.split("/")[-3]
    # env = GymEnvironment(env_id, append_time=True)
    # env  = gym.make(args.env) # Tyna check
    env = gym.make('HalfCheetah-v2')

    print(env.observation_space)

    # load agent
    print("here 2")
    agent_name = os.path.basename(os.path.dirname(args.dir)).split("_")[0]
    # agent_fn = getattr(continuous, agent_name)()
    print("here 3")
    agent_fn = ppo_continuous()
    agent = agent_fn(env)

    ppo_agent = PPOAgent


    agent.load(args.dir)
    lazy_agent = agent.make_lazy_agent(evaluation=not args.train, store_samples=True)



    # reset ExperienceReplayBuffer
    set_replay_buffer(ExperienceReplayBuffer(args.frames + 10, env))

    # set sampler
    sampler = AsyncSampler(env, num_workers=args.num_workers)

    # start recording
    replay_buffer = get_replay_buffer()


    returns = []
    while len(replay_buffer) < args.frames:
        sampler.start_sampling(
            lazy_agent, worker_episodes=1)

        sample_result = sampler.store_samples(timeout=1)
        for sample_info in sample_result.values():
            returns += sample_info["returns"]

    # save return info of the policy
    returns_dict = {"mean": np.mean(returns), "std": np.std(returns)}
    filepath = os.path.join(args.dir, 'demo_return.json')
    with open(filepath, mode='w') as f:
        json.dump(returns_dict, f)


    # save replay buffer
    filepath = os.path.join(args.dir, 'transitions.pkl')
    with open(filepath, mode='wb') as f:
        samples = replay_buffer.get_all_transitions(return_cpprb=True)
        pickle.dump(samples, f)

    print("Transitions (size: {}) is saved at {}".format(
        len(replay_buffer), filepath))




if __name__ == "__main__":
    main()
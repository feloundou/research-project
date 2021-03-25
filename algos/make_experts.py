from agent_utils import Expert
from adabelief_pytorch import AdaBelief

import gym
import safety_gym

from mpi4py import MPI
import sys
import subprocess
import os
import safety_gym
import gym

from neural_nets import MLPActorCritic
from utils import setup_logger_kwargs, mpi_fork

ENV_NAME = 'Safexp-PointGoal1-v0'



def print_hello(rank, size, name):
  msg = "Hello World! I am process {0} of {1} on {2}.\n"
  sys.stdout.write(msg.format(rank, size, name))

def mpi_fork(n, bind_to_core=False):

    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        print("mpi args: ", args)
        print("sys args types partial", sys.argv)
        # print("sys args types full", sys.argv[0])
        print("testing paths: ", os.path.abspath(sys.argv[0]))
        # print("env: ", env)
        subprocess.check_call(args, env=env)
        sys.exit()




# # make expert
# expert = Expert(config_name='wonder',
#                 record_samples=True,
#                 actor_critic=MLPActorCritic,
#                 ac_kwargs=dict(hidden_sizes=[128] * 4),
#                 seed=0,
#                 penalty_init=5e-3)
#
# logger_kwargs = setup_logger_kwargs('STANDARDTEST', 0)
#
# expert.ppo_train(env_fn=lambda: gym.make(ENV_NAME),
#                  epochs=50,
#                  gamma=0.99,
#                  lam=0.98,
#                  steps_per_epoch=5000,
#                  train_pi_iters=100,
#                  pi_lr=3e-4,
#                  train_vf_iters=100,
#                  vf_lr=1e-3,
#                  penalty_lr=5e-3,
#                  cost_lim=10,
#                  clip_ratio=0.2 ,
#                  max_ep_len=1000,
#                  save_every=10,
#                  wandb_write=False,
#                  logger_kwargs=logger_kwargs)

########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print_hello(rank, size, name)

    print("CPUS have been forked")

    exec(open('nn_config.py').read(), globals())

    mpi_fork(10)  # run parallel code with mpi

    CONFIG_LIST2 = CONFIG_LIST

    for configuration in CONFIG_LIST2:
        print(configuration)


        logger_kwargs_BIG = setup_logger_kwargs(configuration['name'], configuration['seed'])


        BIG_EXPERT = Expert(config_name=configuration['name'],
                            record_samples=True,
                            actor_critic=MLPActorCritic,
                            ac_kwargs=dict(hidden_sizes=[configuration['hid']] * configuration['l']),
                            seed=configuration['seed'],
                            penalty_init=5e-3)


        BIG_EXPERT.ppo_train(env_fn=lambda: gym.make(ENV_NAME),
                             epochs=1000,
                             gamma=configuration['gamma'],
                             lam=configuration['lam'],
                             steps_per_epoch=configuration['steps'],
                             train_pi_iters=100,
                             pi_lr=3e-4,
                             train_vf_iters=100,
                             vf_lr=1e-3,
                             penalty_lr=configuration['penalty_lr'],
                             cost_lim=configuration['cost_lim'],
                             clip_ratio=0.2,
                             max_ep_len=1000,
                             save_every=10,
                             wandb_write=False,
                             logger_kwargs=logger_kwargs_BIG)

        print("just finished!")








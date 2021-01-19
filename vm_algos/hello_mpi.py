from mpi4py import MPI
import sys
import subprocess
import os
import safety_gym
import gym

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

if __name__ == "__main__":
  size = MPI.COMM_WORLD.Get_size()
  rank = MPI.COMM_WORLD.Get_rank()
  name = MPI.Get_processor_name()

  print_hello(rank, size, name)

  mpi_fork(20)

  print("Hi, Computer")

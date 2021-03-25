import gym
import safety_gym
import mujoco_py
import sys
import os

print("imported")

modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]

print(allmodules)
print("mujoco path")

#env = gym.make('Safexp-PointGoal1-v0')
print(mujoco_py.__file__)


print("fixed")


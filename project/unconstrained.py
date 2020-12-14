#!
import os
# os.system()

import safety_gym
import gym
from safety_gym.envs.engine import Engine

from gym.envs.registration import register

env = gym.make('Safexp-PointGoal1-v0')

print("hello")

# config = {
#     'robot_base': 'xmls/car.xml',
#     'task': 'push',
#     'observe_goal_lidar': True,
#     'observe_box_lidar': True,
#     'observe_hazards': True,
#     'observe_vases': True,
#     'constrain_hazards': True,
#     'lidar_max_dist': 3,
#     'lidar_num_bins': 16,
#     'hazards_num': 4,
#     'vases_num': 4
# }
#
# env = Engine(config)

register(id='SafexpTestEnvironment-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config})


if __name__ == '__main__':
    print('hey')
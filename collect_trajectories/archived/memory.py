from cpprb import ReplayBuffer
import numpy as np

buffer_size = 256
obs_shape = 3
act_dim = 1
rb = ReplayBuffer(buffer_size,
		  env_dict ={"obs": {"shape": obs_shape},
			         "act": {"shape": act_dim},
			         "rew": {},
			         "next_obs": {"shape": obs_shape},
			         "done": {}})

obs = np.ones(shape=(obs_shape))
act = np.ones(shape=(act_dim))
rew = 0
next_obs = np.ones(shape=(obs_shape))
done = 0

for i in range(500):
    rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)

    if done:
	# Together with resetting environment, call ReplayBuffer.on_episode_end()
	    rb.on_episode_end()

print("printing replay buffer")
print(rb)

batch_size = 32
sample = rb.sample(batch_size)
print(sample)
# sample is a dictionary whose keys are 'obs', 'act', 'rew', 'next_obs', and 'done'

if __name__ == '__main__':
    print("wow")
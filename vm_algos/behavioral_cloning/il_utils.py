import numpy as np
import torch
from cpprb import (ReplayBuffer, PrioritizedReplayBuffer,
                   create_env_dict, create_before_add_func)

from abc import ABC, abstractmethod
import argparse
# import pybullet
# import pybullet_envs
import os
import time
import pickle
import json
import numpy as np

import torch
import logging
from abc import ABC, abstractmethod



class Writer(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_scalar(self, name, value, step="sample_frames",
                   step_value=None, save_csv=False):
        pass

    @abstractmethod
    def add_text(self, name, text, step="sample_frames"):
        pass

    def _get_step_value(self, _type):
        if type(_type) is not str:
            raise ValueError("step must be str")
        if _type == "sample_frames":
            return self.sample_frames
        if _type == "sample_episodes":
            return self.sample_episodes
        if _type == "train_steps":
            return self.train_steps
        return _type


class DummyWriter(Writer):
    def __init__(self):
        self.sample_frames = 0
        self.sample_episodes = 0
        self.train_steps = 0

    def add_scalar(self, name, value, step="sample_frames",
                   step_value=None, save_csv=False):
        pass

    def add_text(self, name, text, step="sample_frames"):
        pass



os.environ["PYTHONWARNINGS"] = 'ignore:semaphore_tracker:UserWarning'

_DEBUG_MODE = False


def enable_debug_mode():
    global _DEBUG_MODE
    print("-----DEBUG_MODE: True-----")
    torch.autograd.set_detect_anomaly(True)
    _DEBUG_MODE = True


def disable_debug_mode():
    global _DEBUG_MODE
    print("-----DEBUG_MODE: False-----")
    _DEBUG_MODE = False


def is_debug_mode():
    global _DEBUG_MODE
    return _DEBUG_MODE


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_device(device):
    global _DEVICE
    _DEVICE = device


def get_device():
    return _DEVICE


_SEED = 0


def set_seed(seed):
    global _SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _SEED = seed
    print("-----SEED: {}-----".format(_SEED))


def call_seed():
    global _SEED
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    return _SEED


_WRITER = DummyWriter()


def set_writer(writer):
    global _WRITER
    _WRITER = writer


def get_writer():
    return _WRITER


_LOGGER = logging.getLogger(__name__)


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


def get_logger():
    return _LOGGER


_REPLAY_BUFFER = None


def set_replay_buffer(replay_buffer):
    global _REPLAY_BUFFER
    _REPLAY_BUFFER = replay_buffer


def get_replay_buffer():
    global _REPLAY_BUFFER
    if _REPLAY_BUFFER is None:
        raise ValueError("replay_buffer is not set")
    return _REPLAY_BUFFER


_ON_POLICY_MODE = False


def enable_on_policy_mode():
    global _ON_POLICY_MODE
    _ON_POLICY_MODE = True
    print("-----ON_POLICY_MODE: {}-----".format(_ON_POLICY_MODE))


def disable_on_policy_mode():
    global _ON_POLICY_MODE
    _ON_POLICY_MODE = False
    print("-----ON_POLICY_MODE: {}-----".format(_ON_POLICY_MODE))


def is_on_policy_mode():
    global _ON_POLICY_MODE
    return _ON_POLICY_MODE


# parameters of NstepExperienceReplay
_NSTEP = 1
_DISCOUNT_FACTOR = 0.95


def set_n_step(n_step, discount_factor=0.95):
    global _NSTEP, _DISCOUNT_FACTOR
    _NSTEP = n_step
    _DISCOUNT_FACTOR = discount_factor
    print("-----N step: {}-----".format(_NSTEP))
    print("-----Discount factor: {}-----".format(_DISCOUNT_FACTOR))


def get_n_step():
    global _NSTEP, _DISCOUNT_FACTOR
    return _NSTEP, _DISCOUNT_FACTOR


_USE_APEX = False


def enable_apex():
    global _USE_APEX
    _USE_APEX = True
    print("-----USE_APEX: {}-----".format(_USE_APEX))


def disable_apex():
    global _USE_APEX
    _USE_APEX = False
    print("-----USE_APEX: {}-----".format(_USE_APEX))


def use_apex():
    global _USE_APEX
    return _USE_APEX



def check_samples(samples, priorities=None):
    states, actions, rewards, next_states, _, _ = samples

    # type check
    assert isinstance(states, State), "Input invalid states type {}. \
        states must be all.environments.State".format(type(states))
    assert isinstance(actions, Action), "Input invalid states type {}. \
            actions must be all.environments.Action".format(type(actions))
    assert isinstance(next_states, State), \
        "Input invalid next_states type {}. next_states must be all.environments.State".format(
        type(next_states))
    assert isinstance(rewards, torch.Tensor), "Input invalid rewards type {}. \
        rewards must be torch.Tensor".format(type(rewards))
    if priorities is not None:
        assert isinstance(priorities, torch.Tensor), "Input invalid priorities type {}. \
            priorities must be torch.Tensor".format(type(priorities))

    # shape check
    assert len(rewards.shape) == 1, \
        "rewards.shape {} must be 'shape == (batch_size)'".format(
            rewards.shape)
    if priorities is not None:
        assert len(priorities.shape) == 1, \
            "priorities.shape {} must be 'shape == (batch_size)'".format(
                priorities.shape)


def check_inputs_shapes(store):
    def retfunc(self, samples, priorities=None):
        if samples.states is None:
            return None
        if is_debug_mode():
            check_samples(samples, priorities=priorities)
        return store(self, samples, priorities=priorities)
    return retfunc

class BaseReplayBuffer(ABC):
    @abstractmethod
    def store(self, states, actions, rewards, next_states):
        """Store the transition in the buffer
        Args:
            states (rlil.environment.State): batch_size x shape
            actions (rlil.environment.Action): batch_size x shape
            rewards (torch.Tensor): batch_size
            next_states (rlil.environment.State): batch_size x shape
        """

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''

    @abstractmethod
    def get_all_transitions(self):
        '''Return all the samples'''

    @abstractmethod
    def clear(self):
        '''Clear replay buffer'''


class BaseBufferWrapper(ABC):
    def __init__(self, buffer):
        self.buffer = buffer

    def store(self, *args, **kwargs):
        self.buffer.store(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        self.buffer.update_priorities(*args, **kwargs)

    def clear(self):
        self.buffer.clear()

    def get_all_transitions(self):
        return self.buffer.get_all_transitions()

    def samples_from_cpprb(self, *args, **kwargs):
        return self.buffer.samples_from_cpprb(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)

class ExperienceReplayBuffer(BaseReplayBuffer):
    '''This class utilizes cpprb.ReplayBuffer'''

    def __init__(self,
                 size, env,
                 prioritized=False, alpha=0.6, beta=0.4, eps=1e-4,
                 n_step=1, discount_factor=0.95):
        """
        Args:
            size (int): The capacity of replay buffer.
            env (rlil.environments.GymEnvironment)
            prioritized (bool): Use prioritized replay buffer if True.
            alpha, beta, eps (float):
                Hyperparameter of PrioritizedReplayBuffer.
                See https://arxiv.org/abs/1511.05952.
            n_step (int, optional):
               Number of steps for Nstep experience replay.
               If n_step > 1, you need to call self.on_episode_end()
               before every self.sample(). The n_step calculation is done
               in LazyAgent objects, not in Agent objects.
            discount_factor (float, optional):
                Discount factor for Nstep experience replay.
        """

        # common
        self._before_add = create_before_add_func(env)
        self.device = get_device()
        env_dict = create_env_dict(env)

        # Nstep
        Nstep = None
        if n_step > 1:
            Nstep = {"size": n_step, "rew": "rew",
                     "next": "next_obs", "gamma": discount_factor}
        self._n_step = n_step

        # PrioritizedReplayBuffer
        self.prioritized = prioritized
        self._beta = beta
        if prioritized:
            self._buffer = PrioritizedReplayBuffer(size, env_dict,
                                                   alpha=alpha, eps=eps,
                                                   Nstep=Nstep)
        else:
            self._buffer = ReplayBuffer(size, env_dict, Nstep=Nstep)

    @check_inputs_shapes
    def store(self, samples, priorities=None):
        """Store the samples in the buffer
        Args:
            Samples(
                states (rlil.environment.State): batch_size x shape
                actions (rlil.environment.Action): batch_size x shape
                rewards (torch.Tensor): batch_size
                next_states (rlil.environment.State): batch_size x shape
                weights: None
                indexes: None
            )
            priorities (torch.Tensor): batch_size
        """

        np_states, np_rewards, np_actions, np_next_states, \
            np_dones, np_next_dones = samples_to_np(samples)

        assert len(np_states) < self._buffer.get_buffer_size(), \
            "The sample size exceeds the buffer size."

        if self.prioritized and (~np_dones).any():
            np_priorities = None if priorities is None \
                else priorities.detach().cpu().numpy()[~np_dones]
            self._buffer.add(
                **self._before_add(obs=np_states[~np_dones],
                                   act=np_actions[~np_dones],
                                   rew=np_rewards[~np_dones],
                                   done=np_next_dones[~np_dones],
                                   next_obs=np_next_states[~np_dones]),
                priorities=np_priorities)

        # if there is at least one sample to store
        if not self.prioritized and (~np_dones).any():
            # remove done==1 by [~np_dones]
            self._buffer.add(
                **self._before_add(obs=np_states[~np_dones],
                                   act=np_actions[~np_dones],
                                   rew=np_rewards[~np_dones],
                                   done=np_next_dones[~np_dones],
                                   next_obs=np_next_states[~np_dones]))

    def sample(self, batch_size):
        '''Sample from the stored transitions'''
        if self.prioritized:
            npsamples = self._buffer.sample(batch_size, beta=self._beta)
        else:
            npsamples = self._buffer.sample(batch_size)
        samples = self.samples_from_cpprb(npsamples)
        return samples

    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''
        if is_debug_mode():
            # shape check
            assert len(td_errors.shape) == 1, \
                "rewards.shape {} must be 'shape == (batch_size)'".format(
                    rewards.shape)
            assert td_errors.device == torch.device("cpu"), \
                "td_errors must be cpu tensors"

        if self.prioritized:
            self._buffer.update_priorities(indexes, td_errors.detach().numpy())

    def get_all_transitions(self, return_cpprb=False):
        npsamples = self._buffer.get_all_transitions()
        if return_cpprb:
            return npsamples
        return self.samples_from_cpprb(npsamples)

    def samples_from_cpprb(self, npsamples, device=None):
        """
        Convert samples generated by cpprb.ReplayBuffer.sample() into
        State, Action, rewards, State.
        Return Samples object.
        Args:
            npsamples (dict of nparrays):
                Samples generated by cpprb.ReplayBuffer.sample()
            device (optional): The device where the outputs are loaded.
        Returns:
            Samples(State, Action, torch.FloatTensor, State)
        """
        device = self.device if device is None else device

        states = State.from_numpy(npsamples["obs"], device=device)
        actions = Action.from_numpy(npsamples["act"], device=device)
        rewards = torch.tensor(npsamples["rew"], dtype=torch.float32,
                               device=device).squeeze()
        next_states = State.from_numpy(
            npsamples["next_obs"], npsamples["done"], device=device)
        if self.prioritized:
            weights = torch.tensor(
                npsamples["weights"], dtype=torch.float32, device=self.device)
            indexes = npsamples["indexes"]
        else:
            weights = torch.ones(states.shape[0], device=self.device)
            indexes = None
        return Samples(states, actions, rewards, next_states, weights, indexes)

    def on_episode_end(self):
        if self._n_step > 1:
            self._buffer.on_episode_end()

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return self._buffer.get_stored_size()

if __name__ == '__main__':
    print("replay replay")

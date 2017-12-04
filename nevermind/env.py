from abc import ABCMeta, abstractmethod
from collections import deque
from random import randint
from typing import Tuple, Sequence

import cv2
import numpy as np
from gym import Env, Wrapper, ObservationWrapper, RewardWrapper
from gym.envs.atari import AtariEnv
from gym.spaces import Box
from gym.wrappers import SkipWrapper
from numpy import ndarray


class NoopRandomStart(Wrapper):
    def _reset(self):
        observation = self.env.reset()
        for _ in range(randint(1, self.max_num_noop)):
            observation, reward, done, info = self.env.step(self.noop_action)

            if done:
                observation = self.env.reset()

        return observation

    def __init__(self, env: Env, max_num_noop: int = 30, noop_action: int = 0):
        super().__init__(env)
        self.noop_action = noop_action
        self.max_num_noop = max_num_noop


class Transpose(ObservationWrapper):
    def __init__(self, new_axis_order: Sequence[int], env: Env):
        super().__init__(env)
        self.new_axis_order = new_axis_order
        e = self.env.observation_space
        if not (isinstance(e, Box) and len(e.shape) == len(new_axis_order)):
            raise Exception('Axes count should match environment box shape dimensionality.')
        self.observation_space = Box(e.low.item(0), e.high.item(0), [e.shape[i] for i in new_axis_order])

    def _observation(self, observation: ndarray):
        swapped = observation.transpose(self.new_axis_order)

        assert self.observation_space.contains(swapped)

        return swapped


class Grayscale(ObservationWrapper):
    def __init__(self, env: Env, weights: Tuple[float] = (0.299, 0.587, 0.114)):
        super().__init__(env)
        self.weights = np.array(weights)
        e = self.env.observation_space
        if not (isinstance(e, Box) and len(e.shape) == 3 and len(self.weights.shape) == 1 and
                        e.shape[-1] == self.weights.shape[0]):
            raise Exception('Last dimension of 2D observation space must match the dimensionality of the weights.')
        self.observation_space = Box(e.low.item(0), e.high.item(0), e.shape[:-1])

    def _observation(self, observation: ndarray):
        grayscaled = (observation * self.weights).sum(-1)
        assert self.observation_space.contains(grayscaled)
        return grayscaled


class Resize(ObservationWrapper):
    def __init__(self, shape: Sequence[int], env: Env):
        super().__init__(env)
        self.shape = shape

        e = self.env.observation_space
        if not (isinstance(e, Box) and len(e.shape) >= 2):
            raise Exception('Observation space must have at least two dimensions.')

        self.observation_space = Box(e.low.item(0), e.high.item(0), shape)

    def _observation(self, observation: ndarray):
        resized = cv2.resize(observation, self.shape)
        assert self.observation_space.contains(resized)
        return resized


class Normalize(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        e = self.env.observation_space
        if not (isinstance(e, Box)):
            raise Exception('Observation space must be a box.')

        self.observation_space = Box(0, 1, e.shape)

    def _observation(self, observation: ndarray):
        low, high = self.env.observation_space.low.item(0), self.env.observation_space.high.item(0)
        normalized = (observation.astype(float) - low) / (high - low)
        assert self.observation_space.contains(normalized)
        return normalized


class ConvolutionWrapper(Wrapper):
    __metaclass__ = ABCMeta

    def _reset(self):
        observation = self.env.reset()

        for _ in range(self.size):
            self.observations_queue.append(observation)

        return self._observation()

    def _step(self, action):
        observation, reward, done, info = self.env.step(action=action)

        self.observations_queue.append(observation)

        return self._observation(), reward, done, info

    def _observation(self):
        convolved = self.convolve(np.array(self.observations_queue))
        assert self.observation_space.contains(convolved)
        return convolved

    @abstractmethod
    def convolve(self, observations: ndarray):
        raise NotImplementedError

    def __init__(self, env: Env, size: int):
        super().__init__(env)
        self.size = size
        self.observations_queue = deque(maxlen=size)


class Maximum(ConvolutionWrapper):
    def convolve(self, observations: ndarray):
        return observations.max(0)

    def __init__(self, env: Env, size: int = 2):
        super().__init__(env, size=size)


class History(ConvolutionWrapper):
    def convolve(self, observations: ndarray):
        return observations

    def __init__(self, env: Env, size: int = 4):
        super().__init__(env, size=size)
        e = self.env.observation_space
        self.observation_space = Box(e.low.item(0), e.high.item(0), (size,) + e.shape)


class ClipReward(RewardWrapper):
    def _reward(self, reward):
        return np.sign(reward)


def atari_env(game: str):
    return ClipReward(Transpose((1, 2, 0), History(SkipWrapper(4)(Resize((84, 84), Normalize(Grayscale(
        Maximum(NoopRandomStart(AtariEnv(game=game, obs_type='image', frameskip=1))))))))))

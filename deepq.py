import abc
from typing import List

import numpy as np
from gym import Env
from gym.spaces import Discrete
from numpy import ndarray

from replay import Experience


class QFunctionApproximation:
    __metaclass__ = abc.ABCMeta

    def __init__(self, env: Env):
        self.env = env
        space = self.env.action_space

        if not isinstance(space, Discrete):
            raise ValueError('Action space must be discrete.')

        self.all_actions = np.array(range(space.n))

    @abc.abstractmethod
    def values_for(self, observations: ndarray, actions: ndarray) -> ndarray:
        raise NotImplementedError("To override.")

    def greedy_action(self, observation: ndarray) -> ndarray:
        values = self.values_for(observations=np.array([observation] * len(self.all_actions)), actions=self.all_actions)

        best_action_indices = np.squeeze(np.argwhere(values == np.amax(values)))

        return self.all_actions[np.random.choice(best_action_indices)]

    @abc.abstractmethod
    def update(self, experiences: List[Experience]):
        raise NotImplementedError("To override.")


class DeepQNetwork(QFunctionApproximation):
    def __init__(self, env: Env):
        super().__init__(env)

    def values_for(self, observations: ndarray, actions: ndarray):
        raise NotImplementedError

    def update(self, experiences: List[Experience]):
        raise NotImplementedError

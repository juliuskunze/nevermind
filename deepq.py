from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Callable, Sequence

import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces import Discrete
from keras import Sequential, losses
from keras.layers import Dense, K
from keras.optimizers import Adam, Optimizer
from numpy import ndarray

from replay import Experience


class ValueFunctionApproximation(ABC):
    def __init__(self, env: Env):
        self.env = env
        space = self.env.action_space

        if not isinstance(space, Discrete):
            raise ValueError('Action space must be discrete.')

    @abstractmethod
    def all_action_values_for(self, observations: ndarray) -> ndarray:
        """Returns values for all possible actions for all observations in the provided batch."""
        raise NotImplementedError

    def all_greedy_actions(self, observation: ndarray) -> ndarray:
        values = self.all_action_values_for(observations=np.array([observation]))[0]

        return np.squeeze(np.argwhere(values == np.amax(values)), axis=-1)

    def greedy_action(self, observation: ndarray) -> ndarray:
        return np.random.choice(self.all_greedy_actions(observation))

    @abstractmethod
    def update(self, experiences: List[Experience]):
        raise NotImplementedError


def mlp(input_shape: Tuple[int], num_outputs: int, hidden_layers_sizes: Sequence[int] = (64,)):
    return Sequential(layers=[Dense(size, activation='relu', input_shape=input_shape if index == 0 else [None])
                              for index, size in enumerate(hidden_layers_sizes)] +
                             [Dense(num_outputs)])


def mean_huber_loss(predictions, labels):
    return tf.losses.huber_loss(labels=labels, predictions=predictions)


class DeepQNetwork(ValueFunctionApproximation):
    def __init__(self, env: Env, architecture: Callable[[], Sequential] = None,
                 target_model_update_period: int = 500,
                 clip_error: bool = False,
                 optimizer: Optimizer = Adam(lr=1e-3),
                 discount_factor: float = 1.):
        super().__init__(env)

        self.clip_error = clip_error

        if architecture is None:
            architecture = lambda: mlp(input_shape=self.env.observation_space.shape,
                                       num_outputs=self.env.action_space.n)

        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.target_model_update_period = target_model_update_period

        self.model, self.model_function = self.create_model_and_function(architecture)
        self.target_model, self.target_model_function = self.create_model_and_function(architecture)

        self.update_target_model()

    def create_model_and_function(self, architecture: Callable[[], Sequential]):
        result = architecture()
        result.compile(optimizer=self.optimizer, loss=mean_huber_loss if self.clip_error else losses.mean_squared_error)
        return result, K.function(result.inputs, result.outputs)

    def update_target_model(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)
        self.steps_since_last_target_model_update = 0
        print('Updated target model.')

    def all_action_values_for(self, observations: ndarray):
        return self.target_model_function([observations])[0]

    def update(self, experiences: List[Experience]):
        def null_to_nan(observation):
            return np.full(self.env.observation_space.shape, np.nan) if observation is None else observation

        observations = np.array([e.observation for e in experiences])
        next_observations = np.array([null_to_nan(e.next_observation) for e in experiences])
        actions = np.array([e.action for e in experiences])

        next_value_estimates = self.all_action_values_for(next_observations)

        # The target is set to the prediction for all actions but the one have a target value for from the experience.
        # Therefore, the error is only non-zero for one particular action and we can just sum to get this error.
        target_values = [e.reward + self.discount_factor * (
            0 if e.is_terminal else max(next_value_estimates[i])) for i, e in enumerate(experiences)]

        targets = self.model_function([observations])[0]
        for target, action, target_value in zip(targets, actions, target_values):
            target[action] = target_value

        loss = self.model.train_on_batch(observations, targets)

        self.steps_since_last_target_model_update += 1
        if self.steps_since_last_target_model_update == self.target_model_update_period:
            self.update_target_model()

        return loss

    def save(self, file: Path):
        self.target_model.save(str(file))

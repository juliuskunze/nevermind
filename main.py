from typing import Callable, List

import gym
import numpy as np
from gym import Env
from numpy import ndarray

from deepq import QFunctionApproximation
from replay import Experience, ReplayBuffer


def run(env: Env, policy: Callable[[ndarray], ndarray], render=True):
    while True:
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            observation, reward, done, info = env.step(policy(observation))
            episode_reward += reward
        print(f'Episode reward: {episode_reward}')


def run_greedy(q: QFunctionApproximation, render=True):
    run(q.env, lambda observation: q.greedy_action(observation), render=render)


def train(q: QFunctionApproximation, batch_size: int = 32, buffer_size: int = 100, render=True):
    buffer = ReplayBuffer(size=buffer_size)
    env = q.env

    while True:
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()

            action = q.greedy_action(observation)
            new_observation, reward, done, info = env.step(action)

            buffer.add(Experience(observation, action, reward, new_observation))

            observation = new_observation

            episode_reward += reward
            q.update(experiences=buffer.sample(num=batch_size))

        print(f'Episode reward: {episode_reward}')


class ZeroQ(QFunctionApproximation):
    def __init__(self, env: Env):
        super().__init__(env)

    def values_for(self, observations: ndarray, actions: ndarray):
        return np.array([0] * observations.shape[0])

    def update(self, experiences: List[Experience]):
        pass


def main():
    env = gym.make('CartPole-v0')

    train(ZeroQ(env))


if __name__ == '__main__':
    main()

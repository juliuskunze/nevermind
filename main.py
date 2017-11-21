from pathlib import Path
from typing import Callable, List

import gym
import numpy as np
from gym import Env
from numpy import ndarray

from deepq import ValueFunctionApproximation, DeepQNetwork
from plot import plot_training_summary, plot_cartpole_value_function
from replay import Experience
from train import train, timestamp, PeriodicTrainingCallback, TrainingSummary


def save_cartpole_q_plot_callback(period: int = 10000,
                                  directory: Path = None,
                                  show_advantage=False):
    name = 'advantage' if show_advantage else 'value'

    if directory is None:
        directory = Path('data') / 'plots' / name / timestamp()

    def save(summary: TrainingSummary):
        if directory is not None:
            directory.mkdir(exist_ok=True, parents=True)

        save_to_file = None if directory is None else \
            directory / f'{name}_step{summary.timestep}'

        plot_cartpole_value_function(summary.q, save_to_file=save_to_file, show_advantage=show_advantage)

    return PeriodicTrainingCallback(action=save, period=period)


def run(env: Env, policy: Callable[[ndarray], ndarray], render=True, episodes=50):
    episode_rewards: List[float] = []
    for _ in range(episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            observation, reward, done, info = env.step(policy(observation))
            episode_reward += reward
        print(f'Episode reward: {episode_reward}')
        episode_rewards.append(episode_reward)

    return episode_rewards


def run_greedy(q: ValueFunctionApproximation, render=True):
    run(q.env, lambda observation: q.greedy_action(observation), render=render)


class ZeroQ(ValueFunctionApproximation):
    def __init__(self, env: Env):
        super().__init__(env)

    def all_action_values_for(self, observations: ndarray):
        return np.zeros((observations.shape[0], self.env.action_space.n))

    def update(self, experiences: List[Experience]):
        pass


def main():
    env = gym.make('CartPole-v0')

    def is_solved(summary: TrainingSummary):
        return len(summary.episode_rewards) >= 100 and np.average(summary.episode_rewards[-100:]) >= 199

    summary = train(DeepQNetwork(env),
                    callbacks=[PeriodicTrainingCallback.save_dqn(),
                               save_cartpole_q_plot_callback(),
                               save_cartpole_q_plot_callback(show_advantage=True)],
                    is_solved=is_solved)
    plot_training_summary(summary)


if __name__ == '__main__':
    main()

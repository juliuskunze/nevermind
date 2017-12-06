import random
from math import inf
from pathlib import Path
from typing import Callable, Sequence, List

import numpy as np

from nevermind.deepq import ValueFunctionApproximation, DeepQNetwork
from nevermind.replay import ReplayBuffer, Experience


def linearly_decreasing_exploration(initial_exploration: float, decrease_start: int, decrease_timesteps: int,
                                    final_exploration: float):
    def exploration(timestep: float):
        decrease_progress = (timestep - decrease_start) / decrease_timesteps

        return initial_exploration if decrease_progress <= 0 else (
            final_exploration if decrease_progress >= 1 else
            initial_exploration * (1 - decrease_progress) + final_exploration * decrease_progress)

    return exploration


class TrainingSummary:
    def __init__(self, q: ValueFunctionApproximation, num_timesteps: int, timestep: int = 0):
        self.num_timesteps = num_timesteps
        self.q = q
        self.timestep = timestep
        self.returns: List[float] = []
        self.losses: List[float] = []
        self.exploration_rates: List[float] = []
        self.episode_lengths: List[int] = []
        self.buffer_sizes: List[int] = []

    def average_return(self, num_last_episodes: int):
        return np.average(self.returns[-num_last_episodes:]) if len(self.returns) > 0 else -inf

    @property
    def last_return(self):
        return self.returns[-1] if len(self.returns) > 0 else -inf

    @property
    def last_exploration_rate(self):
        return self.exploration_rates[-1]

    def __str__(self):
        return f'timestep {self.timestep} ({100 * self.timestep / self.num_timesteps:.2f}%), ' \
               f'episode #{len(self.returns)} return: {self.last_return} (recent average: {self.average_return(num_last_episodes=100):.2f}), ' \
               f'exploration rate: {self.last_exploration_rate:.2f}'


class PeriodicTrainingCallback:
    def __init__(self, period: int, action: Callable[[TrainingSummary], None]):
        self.action = action
        self.period = period

    def __call__(self, summary: TrainingSummary):
        self.action(summary)

    @staticmethod
    def save_dqn_if_improved(period: int = 10000, file: Path = None, num_last_episodes_for_average=100):
        class SaveCallback(PeriodicTrainingCallback):
            def __init__(self):
                self.last_average_return = -inf

                def save_if_improved(summary: TrainingSummary):
                    average_return = summary.average_return(num_last_episodes_for_average)
                    if average_return > self.last_average_return:
                        assert isinstance(summary.q, DeepQNetwork)
                        summary.q.save(file)
                        print(f'Recent return average improved from {self.last_average_return} to {average_return}, '
                              f'saving model to {file}.')
                        self.last_average_return = average_return

                super().__init__(period, action=save_if_improved)

        return SaveCallback()

    @staticmethod
    def render():
        return PeriodicTrainingCallback(action=lambda summary: summary.q.env.render(), period=1)

    @staticmethod
    def print_progress(period=1000):
        return PeriodicTrainingCallback(action=lambda summary: print(summary), period=period)


def train(q: ValueFunctionApproximation,
          batch_size: int = 32,
          replay_buffer_size: int = 50000,
          num_timesteps: int = 100000,
          learning_starts: int = 1000,
          final_exploration: float = .02,
          exploration_time_share: float = .1,
          exploration_by_timestep: Callable[[float], float] = None,
          callbacks: Sequence[PeriodicTrainingCallback] = (PeriodicTrainingCallback.print_progress(),),
          is_solved: Callable[[TrainingSummary], bool] = lambda s: False):
    if exploration_by_timestep is None:
        exploration_by_timestep = linearly_decreasing_exploration(
            initial_exploration=1.,
            decrease_start=learning_starts,
            decrease_timesteps=int(num_timesteps * exploration_time_share),
            final_exploration=final_exploration)

    summary = TrainingSummary(q=q, num_timesteps=num_timesteps)
    buffer = ReplayBuffer(size=replay_buffer_size)
    env = q.env

    while summary.timestep < num_timesteps:
        observation = env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        while True:
            exploration_rate = exploration_by_timestep(summary.timestep)
            summary.exploration_rates.append(exploration_rate)

            for callback in callbacks:
                if summary.timestep % callback.period == 0:
                    callback(summary)

            explore = random.random() < exploration_rate
            action = env.action_space.sample() if explore else q.greedy_action(observation)
            next_observation, reward, done, info = (None, 0., True, None) if done else env.step(int(action))

            experience = Experience(observation, action, reward, next_observation)
            buffer.add(experience)
            summary.buffer_sizes.append(len(buffer))

            observation = next_observation

            episode_return += reward
            episode_length += 1
            if summary.timestep >= learning_starts:
                experiences = buffer.sample(num=batch_size)
                loss = q.update(experiences=experiences)
                summary.losses.append(loss)
            summary.timestep += 1

            if experience.is_terminal:
                break

        summary.returns.append(episode_return)
        summary.episode_lengths.append(episode_length)

        if is_solved(summary):
            for callback in callbacks:
                callback(summary)
            print('Environment solved.')
            break

    return summary

import random
from pathlib import Path
from time import strftime
from typing import Callable, Sequence, List

import numpy as np

from deepq import ValueFunctionApproximation, DeepQNetwork
from replay import ReplayBuffer, Experience


def linearly_decreasing_exploration(initial_exploration: float, timesteps: int, final_exploration: float = .1):
    def exploration(timestep: float):
        progress = timestep / timesteps

        return final_exploration if progress >= 1 else \
            initial_exploration * (1 - progress) + final_exploration * progress

    return exploration


def timestamp() -> str:
    return strftime("%Y%m%d-%H%M%S")


class TrainingContext:
    def __init__(self, timestep: int, q: ValueFunctionApproximation):
        self.q = q
        self.timestep = timestep


class PeriodicTrainingCallback:
    def __init__(self, period: int, action: Callable[[TrainingContext], None]):
        self.action = action
        self.period = period

    def __call__(self, context: TrainingContext):
        self.action(context)

    @staticmethod
    def save_dqn(period: int = 10000, directory: Path = Path('data') / 'models' / timestamp()):
        def save(context: TrainingContext):
            if directory is not None:
                directory.mkdir(exist_ok=True, parents=True)

            save_to_file = None if directory is None else \
                directory / f'step{context.timestep}.model'

            assert isinstance(context.q, DeepQNetwork)
            context.q.save(save_to_file)

        return PeriodicTrainingCallback(action=save, period=period)

    @staticmethod
    def render():
        return PeriodicTrainingCallback(action=lambda context: context.q.env.render(), period=1)


class TrainingSummary:
    def __init__(self, episode_rewards: List[float] = list(), losses: List[float] = list()):
        self.losses = losses
        self.episode_rewards = episode_rewards


def train(q: ValueFunctionApproximation,
          batch_size: int = 32,
          buffer_size: int = 50000,
          num_timesteps: int = 100000,
          learning_starts: int = 1000,
          exploration_by_timestep: Callable[[float], float] = None,
          callbacks: Sequence[PeriodicTrainingCallback] = ()):
    if exploration_by_timestep is None:
        exploration_by_timestep = linearly_decreasing_exploration(initial_exploration=1.,
                                                                  timesteps=50000,  # int(num_timesteps * .1),
                                                                  final_exploration=.02)

    summary = TrainingSummary()
    buffer = ReplayBuffer(size=buffer_size)
    env = q.env
    timestep = 0

    while timestep < num_timesteps:
        observation = env.reset()
        done = False
        episode_reward = 0
        episode_action_counts = np.zeros([env.action_space.n])
        while not done:
            for callback in callbacks:
                if timestep % callback.period == 0:
                    callback(TrainingContext(timestep, q))

            explore = random.random() < exploration_by_timestep(timestep)
            action = env.action_space.sample() if explore else q.greedy_action(observation)
            next_observation, reward, done, info = env.step(int(action))

            buffer.add(Experience(observation, action, reward, next_observation))

            observation = next_observation

            episode_reward += reward
            if timestep >= learning_starts:
                experiences = buffer.sample(num=batch_size)
                loss = q.update(experiences=experiences)
                summary.losses.append(loss)
            timestep += 1
            episode_action_counts[action] += 1

        print(f'Episode reward: {episode_reward}, action counts: {episode_action_counts}')
        summary.episode_rewards.append(episode_reward)

    return summary

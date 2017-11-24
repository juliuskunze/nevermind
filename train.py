import random
from pathlib import Path
from typing import Callable, Sequence, List

from deepq import ValueFunctionApproximation, DeepQNetwork
from replay import ReplayBuffer, Experience


def linearly_decreasing_exploration(initial_exploration: float, decrease_start: int, decrease_timesteps: int,
                                    final_exploration: float):
    def exploration(timestep: float):
        decrease_progress = (timestep - decrease_start) / decrease_timesteps

        return initial_exploration if decrease_progress <= 0 else (
            final_exploration if decrease_progress >= 1 else
            initial_exploration * (1 - decrease_progress) + final_exploration * decrease_progress)

    return exploration


class TrainingSummary:
    def __init__(self,
                 q: ValueFunctionApproximation,
                 timestep: int = 0,
                 returns: List[float] = list(),
                 losses: List[float] = list(),
                 exploration_rates: List[float] = list(),
                 episode_lengths: List[int] = list(),
                 buffer_size: List[int] = list()):
        self.buffer_sizes = buffer_size
        self.episode_lengths = episode_lengths
        self.exploration_rates = exploration_rates
        self.q = q
        self.timestep = timestep
        self.losses = losses
        self.returns = returns


class PeriodicTrainingCallback:
    def __init__(self, period: int, action: Callable[[TrainingSummary], None]):
        self.action = action
        self.period = period

    def __call__(self, summary: TrainingSummary):
        self.action(summary)

    @staticmethod
    def save_dqn(period: int = 10000, directory: Path = None):
        def save(summary: TrainingSummary):
            if directory is not None:
                directory.mkdir(exist_ok=True, parents=True)

            save_to_file = None if directory is None else \
                directory / f'step{summary.timestep}.model'

            assert isinstance(summary.q, DeepQNetwork)
            summary.q.save(save_to_file)

        return PeriodicTrainingCallback(action=save, period=period)

    @staticmethod
    def render():
        return PeriodicTrainingCallback(action=lambda context: context.q.env.render(), period=1)


def train(q: ValueFunctionApproximation,
          batch_size: int = 32,
          replay_buffer_size: int = 50000,
          num_timesteps: int = 100000,
          learning_starts: int = 1000,
          final_exploration: float = .02,
          exploration_time_share: float = .1,
          exploration_by_timestep: Callable[[float], float] = None,
          callbacks: Sequence[PeriodicTrainingCallback] = (),
          is_solved: Callable[[TrainingSummary], bool] = lambda s: False):
    if exploration_by_timestep is None:
        exploration_by_timestep = linearly_decreasing_exploration(
            initial_exploration=1.,
            decrease_start=learning_starts,
            decrease_timesteps=int(num_timesteps * exploration_time_share),
            final_exploration=final_exploration)

    summary = TrainingSummary(q=q)
    buffer = ReplayBuffer(size=replay_buffer_size)
    env = q.env

    while summary.timestep < num_timesteps:
        observation = env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        while True:
            for callback in callbacks:
                if summary.timestep % callback.period == 0:
                    callback(summary)

            exploration_rate = exploration_by_timestep(summary.timestep)
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
            summary.exploration_rates.append(exploration_rate)

            if experience.is_terminal:
                break

        print(f'Return #{len(summary.returns)}: {episode_return}, exploration rate: {exploration_rate:.2f}')
        summary.returns.append(episode_return)
        summary.episode_lengths.append(episode_length)

        if is_solved(summary):
            print('Environment solved.')
            break

    return summary

from pathlib import Path
from time import strftime
from typing import Callable, List

import gym
import numpy as np
import tensorflow as tf
from gym import Env
from keras import optimizers, Sequential
from keras.layers import Conv2D, Dense, Flatten
from numpy import ndarray

from nevermind.deepq import ValueFunctionApproximation, DeepQNetwork
from nevermind.plot import plot_training_summaries, plot_cartpole_value_function
from nevermind.train import train, PeriodicTrainingCallback, TrainingSummary


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


def save_cartpole_q_plot_callback(period: int = 10000,
                                  run_name: str = timestamp(),
                                  directory: Path = None,
                                  show_advantage=False):
    name = 'advantage' if show_advantage else 'value'

    if directory is None:
        directory = Path('data') / 'plots' / 'CartPole-v0' / name / run_name

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


def summary_file(env_name: str, run_name: str):
    return Path('data') / 'plots' / env_name / 'summary' / f'summary{run_name}'


def save_callback(env_name: str, run_name: str):
    return PeriodicTrainingCallback.save_dqn_if_improved(file=Path('data') / 'models' / env_name / f'{run_name}.model')


def average_return_above(minimum_average_return: float, num_last_episodes: int = 100):
    def is_stable(summary: TrainingSummary):
        return len(summary.returns) >= num_last_episodes and summary.average_return(
            num_last_episodes) >= minimum_average_return

    return is_stable


def train_cartpole(num_runs=1, render=True):
    run_name = timestamp()
    env_name = 'CartPole-v0'
    summaries = [train(DeepQNetwork(gym.make(env_name)),
                       callbacks=[PeriodicTrainingCallback.print_progress(),
                                  save_callback(env_name, run_name=name),
                                  save_cartpole_q_plot_callback(run_name=name),
                                  save_cartpole_q_plot_callback(run_name=name, show_advantage=True)] +
                                 ([PeriodicTrainingCallback.render()] if render else []),
                       is_solved=average_return_above(195)) for i, name in
                 enumerate([f'{run_name}{f"-{i}" if i > 0 else ""}' for i in range(num_runs)])]

    plot_training_summaries(summaries, summary_file(env_name, run_name=run_name))


def train_lunar_lander(num_runs=1, render=True):
    run_name = timestamp()
    env_name = 'LunarLander-v2'
    summaries = [train(DeepQNetwork(gym.make(env_name), target_model_update_period=10000),
                       replay_buffer_size=100000,
                       num_timesteps=5000000,
                       callbacks=[PeriodicTrainingCallback.print_progress(),
                                  save_callback(env_name, run_name=f'{run_name}{f"-{i}" if i > 0 else ""}')] +
                                 ([PeriodicTrainingCallback.render()] if render else []),
                       is_solved=average_return_above(200)) for i in range(num_runs)]
    plot_training_summaries(summaries, summary_file(env_name, run_name=run_name))


def train_mountain_car(num_runs=1, render=True):
    run_name = timestamp()
    env_name = 'MountainCar-v0'
    summaries = [train(DeepQNetwork(gym.make(env_name)),
                       callbacks=[PeriodicTrainingCallback.print_progress(),
                                  save_callback(env_name, run_name=f'{run_name}{f"-{i}" if i > 0 else ""}')] +
                                 ([PeriodicTrainingCallback.render()] if render else []),
                       is_solved=average_return_above(-110)) for i in range(num_runs)]
    plot_training_summaries(summaries, summary_file(env_name, run_name=run_name))


def train_atari(game_name='breakout', num_runs: int = 1, render: bool = True, num_timesteps=10000000):
    from nevermind.env import atari_env

    run_name = timestamp()
    env = atari_env(game_name)

    def architecture():
        return Sequential([
            Conv2D(32, (8, 8), strides=(4, 4), input_shape=(84, 84, 4), activation='relu'),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(env.action_space.n)
        ])

    optimizer = optimizers.TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=2.5e-4, momentum=.95, epsilon=.01))

    summary = [train(DeepQNetwork(env, architecture=architecture, optimizer=optimizer,
                                  target_model_update_period=10000,
                                  discount_factor=.99,
                                  clip_error=True,
                                  transpose_observations=lambda x: np.array(x).transpose((0, 2, 3, 1))),
                     num_timesteps=num_timesteps,
                     replay_buffer_size=1000000,
                     final_exploration=.1,
                     exploration_time_share=1000000 / num_timesteps,
                     learning_starts=50000,
                     callbacks=[PeriodicTrainingCallback.print_progress(),
                                save_callback(game_name, run_name=f'{run_name}{f"-{i}" if i > 0 else ""}')] +
                               ([PeriodicTrainingCallback.render()] if render else [])) for i in range(num_runs)]
    plot_training_summaries(summary, summary_file(game_name, run_name=run_name))

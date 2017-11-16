from typing import Any, Callable

import gym
from gym import Env


def random_policy(env: Env):
    return lambda observation: env.action_space.sample()


def run(env: Env, policy: Callable[[Any], Any], render=True):
    while True:
        run_episode(env, policy, render=render)


def run_episode(env: Env, policy: Callable[[Any], Any], render=True):
    observation = env.reset()
    done = False
    episode_reward = 0
    while not done:
        if render:
            env.render()
        observation, reward, done, info = env.step(policy(observation))
        episode_reward += reward
    print(f'Episode reward: {episode_reward}')


def main():
    env = gym.make('CartPole-v0')

    run(env, random_policy(env))


if __name__ == '__main__':
    main()

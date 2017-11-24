from math import pi
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from deepq import ValueFunctionApproximation
from train import TrainingSummary


def save_or_show(fig: Figure, save_to_file: Optional[Path]):
    if save_to_file is None:
        plt.show()
    else:
        Path(save_to_file.parent).mkdir(exist_ok=True, parents=True)
        plt.savefig(str(save_to_file))
        plt.close(fig)


def plot_training_summary(summary: TrainingSummary, save_to_file: Path = None):
    fig, (ax_episode_reward, ax_episode_length, ax_value_loss, ax_exploration_rate, ax_buffer_size) = \
        plt.subplots(nrows=5, figsize=(8, 20))
    fig.suptitle('Training summary')

    ax_episode_reward.set_ylabel('return')
    ax_episode_reward.set_xlabel('episode')
    ax_episode_reward.plot(summary.returns)

    ax_episode_length.set_ylabel('episode length')
    ax_episode_length.set_xlabel('episode')
    ax_episode_length.plot(summary.episode_lengths)

    ax_exploration_rate.set_ylabel('exploration')
    ax_exploration_rate.set_xlabel('timestep')
    ax_exploration_rate.plot(summary.exploration_rates)

    ax_value_loss.set_ylabel('value prediction loss')
    ax_value_loss.set_xlabel('timestep')
    ax_value_loss.plot(summary.losses)

    ax_buffer_size.set_ylabel('buffer size')
    ax_buffer_size.set_xlabel('timestep')
    ax_buffer_size.plot(summary.buffer_sizes)

    save_or_show(fig, save_to_file)


def plot_cartpole_value_function(q: ValueFunctionApproximation, save_to_file: Path = None, show_advantage=False):
    max_x = 2.4
    max_xdot = 3.
    max_θ = 12 * pi / 180
    max_θdot = 3.

    num_x = 5
    num_xdot = 5
    num_θ = 5
    num_θdot = 5

    observations = np.array([[[[[x, xdot, θ, θdot]
                                for x in np.linspace(-max_x, max_x, num=num_x)]
                               for θ in np.linspace(-max_θ, max_θ, num=num_θ)]
                              for xdot in np.linspace(-max_xdot, max_xdot, num=num_xdot)]
                             for θdot in np.linspace(-max_θdot, max_θdot, num=num_θdot)])

    values = np.reshape(q.all_action_values_for(observations=np.reshape(observations, [-1, 4])),
                        list(observations.shape[:-1]) + [q.env.action_space.n])

    fig, axes = plt.subplots(nrows=num_θdot, ncols=num_xdot * 2, sharex=True, sharey=True, figsize=(15, 10))

    name = 'advantage' if show_advantage else 'value'

    fig.canvas.set_window_title(f'cartpole_{name}_function')
    fig.suptitle(f'Cartpole {name} function')

    if show_advantage:
        values -= np.repeat(np.expand_dims(np.average(values, axis=-1), 5), axis=-1, repeats=q.env.action_space.n)

    min_value, max_value = np.min(values), np.max(values)

    for θdot_index, row in enumerate(axes):
        for column_index, ax in enumerate(row):
            xdot_index, action = divmod(column_index, 2)

            _, xdot, _, θdot = observations[θdot_index][xdot_index][0][0]

            ax.set_title(f'{"←" if action == 0 else "→"} $\\dot{{x}}$={xdot} $\\dot{{θ}}$={θdot}', fontsize=5)

            if θdot_index == len(axes) - 1:
                ax.set_xlabel('x')

            if column_index == 0:
                ax.set_ylabel('θ')

            v = values[θdot_index][xdot_index]

            im = ax.imshow(v[:, :, action], extent=[-max_x, max_x, -max_θ, max_θ], aspect='auto', vmin=min_value,
                           vmax=max_value, cmap='seismic' if show_advantage else 'inferno')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=name)

    save_or_show(fig, save_to_file)

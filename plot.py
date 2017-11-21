from math import pi
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from deepq import ValueFunctionApproximation
from train import TrainingSummary, timestamp


def plot_training_summary(summary: TrainingSummary,
                          save_to_file: Path = Path('data') / 'summary' / f'summary{timestamp()}'):
    fig, (ax_reward, ax_loss) = plt.subplots(nrows=2)

    ax_reward.set_title("Reward")
    ax_reward.plot(range(len(summary.episode_rewards)), summary.episode_rewards)

    ax_loss.set_title("Loss")
    ax_loss.plot(range(len(summary.losses)), summary.losses)

    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(str(save_to_file))


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
                           vmax=max_value)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=name)

    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(str(save_to_file))

"""Visualization helper functions
"""
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt

from ..data_loader.config_loader import ColocationConfig
from ..tasks import ground_truth

FONT = {'family': 'monospace', 'size': '8'}


def plot_text(axes: plt.Axes, text: str):
    """Plot text on an axes

    Args:
        axes (plt.Axes): an axes object
        text (str): the text
    """

    axes.axis('off')
    axes.grid('off')
    axes.text(x=0, y=0, s=text, horizontalalignment='left', fontdict=FONT)


def plot_histo(axes: plt.Axes, data, bins=None, num_bins=50):
    """plot a histogram with probability density curve

    Args:
        axes (plt.Axes): axes object
        data (np.ndarray): an np array to be plotted
    """
    if bins is None:
        _, bins, _ = axes.hist(data, bins=num_bins)
    else:
        axes.hist(data, bins=bins)
    axes.set_ylabel('Frequency')

    return bins


def plot_box(axes: plt.Axes, data):
    """plot a box diagram
    """
    axes.boxplot(data)


def plot_1d(axes: plt.Axes, data):
    """plot 1D array
    """
    axes.plot(data)


def plot_fitness_accuracy(config: ColocationConfig, cache):
    """Plot the fitness and accuracy
    """
    fig: plt.Figure = plt.figure(figsize=(6, 4))
    plt.title('Correlational Score for {}'.format(config.job_name))
    axes = plt.subplot(121)
    axes.set_title("Correlational Score")
    axes.plot(cache['best_fitness'])
    axes = plt.subplot(122)
    axes.set_title("Accuracy")
    axes.plot(cache['accuracies'])
    plt.tight_layout()
    if config.save_figure:
        fig.savefig(config.base_file_name + 'fitness_accuracy.png', dpi=300)


def plot_fitness_density(config: ColocationConfig, cache):
    """plot the fitness density
    """
    truth_fitness, _, _ = ground_truth.run(config)

    fig: plt.Figure = plt.figure(figsize=(6, 4))
    plt.title('Overall Correlational Score Density')
    axes = plt.subplot(111)
    bins = plot_histo(axes, np.array(cache['fitnesses']).reshape(-1))
    axes.axvline(truth_fitness, color='red')
    plt.tight_layout()
    if config.save_figure:
        fig.savefig(config.base_file_name + 'fitness_density.png', dpi=300)

    fig2: plt.Figure = plt.figure(figsize=(4, 10))
    plt.title('Correlational Score Density Changing over Time')
    for i in range(5):
        axes = plt.subplot(5, 1, i + 1)
        iteration = i * ((config.max_iteration - 1) // 4)
        axes.set_title('Snapshot of iteration {}'.format(iteration + 1))
        plot_histo(axes,
                   np.array(cache['fitnesses'][iteration]).reshape(-1), bins)
        axes.axvline(truth_fitness, color='red')
        plt.tight_layout()
    if config.save_figure:
        fig2.savefig(
            config.base_file_name + 'fitness_density_over_time.png', dpi=300)


def plot_cache(cache: dict, config: ColocationConfig):
    """Plot a cache dictionary
    """
    if config.plot_fitness_accuracy:
        plot_fitness_accuracy(config, cache)

    if config.plot_fitness_density:
        plot_fitness_density(config, cache)


@contextmanager
def visualizing(*,
                nrows=1,
                ncols=1,
                figsize=(6, 4),
                dpi=300,
                filename='',
                **kwargs):
    """Visualization figure
    """

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    yield axes
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi)

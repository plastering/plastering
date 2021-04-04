"""Exhaust all room combinations
"""

import csv

import numpy as np

from . import strict_ga
from ..data_loader.config_loader import ColocationConfig
from ..utils import visualization


def run(config: ColocationConfig):
    """Run an exhaustive search on all consecutive room

    Args:
        config (ColocationConfig): configuration
    """

    table = []

    for i in range(config.total_room_count - config.room_count + 1):
        config_local = config.copy()
        config_local.selected_rooms = list(range(i, i + config.room_count))
        config_local.visualize = False
        config_local.plot_fitness_accuracy = False
        config_local.plot_fitness_density = False

        print("Rooms {}".format(config_local.selected_rooms))
        best_fitness, best_accuracy, _ = strict_ga.run(config_local)
        table.append([config_local.selected_rooms, best_fitness, best_accuracy])

    print('')
    print('{} consecutive rooms. {} groups checked. '.format(
        config.room_count, config.total_room_count - config.room_count + 1))
    print('average correlational score = {}'.format(np.mean([f for _, f, _ in table])))
    print('average accuracy = {}'.format(np.mean([a for _, _, a in table])))

    print("Saving stats... ")

    # Write to csv file
    with open(
            config.base_file_name + 'room_combination_table.csv', 'w',
            newline='') as file:
        writer = csv.writer(file)
        for row in table:
            writer.writerow(row)

    with visualization.visualizing(
            figsize=(8, 4),
            nrows=2,
            filename=str(config.join_name('room_accuracy.png')),
            sharex=True) as (ax1, ax2):
        x_axis = list(range(len(table)))
        ax1.scatter(x_axis, [f for _, f, _ in table])
        ax1.set_title('fitness')
        ax2.scatter(x_axis, [a for _, _, a, in table])
        ax2.set_title('accuracy')
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels([l for l, _, _ in table], rotation=90)

    return None, None, None

"""Consecutive rooms
"""

import csv

import numpy as np

from . import strict_ga
from ..data_loader.config_loader import ColocationConfig


def run(config: ColocationConfig):
    """run
    """
    table = np.zeros((0, 3), dtype=float)

    config.visualize = False
    config.verbose = False
    config.plot_fitness_accuracy = False
    config.plot_fitness_density = False

    file = config.join_name('consecutive_room.csv').open('w', newline='')
    writer = csv.writer(file)


    for room_count in range(2, 52):
        group_count = config.total_room_count - room_count + 1
        local_table = np.zeros((group_count, 3), dtype=float)

        for i in range(group_count):
            config_local = config.copy()
            config_local.selected_rooms = list(range(i, i + room_count))
            config_local.room_count = room_count
            best_fitness, best_accuracy, _ = strict_ga.run(config_local)
            writer.writerow([room_count, best_fitness, best_accuracy])

            print('Room Count [{}] Group[{}/{}]: {}'.format(room_count, i, group_count,
                                                            best_accuracy))

        table = np.concatenate((table, local_table))

    file.close()

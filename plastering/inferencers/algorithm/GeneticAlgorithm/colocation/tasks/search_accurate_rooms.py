"""Search for rooms where maximizing correlational coefficient leads to
accurate clustering
"""
import multiprocessing
import csv
import sys
from copy import deepcopy
import numpy as np
from ..data_loader import config_loader
from . import strict_ga


def _choose_random_room(total_room_count: int, target_room_count: int):
    return np.random.choice(
        total_room_count, size=target_room_count, replace=False)


_COUNTER_ = multiprocessing.Value('i', 0, lock=True)


def _eval_room_config(config: config_loader.ColocationConfig):
    with _COUNTER_.get_lock():
        _COUNTER_.value += 1
        sys.stderr.write(
            str(_COUNTER_.value) + '/' + str(config.searching_count) + '\n')
    _, accuracy, _ = strict_ga.run(config)

    return accuracy


def run(config: config_loader.ColocationConfig):
    """Search for accurate room groups

    Args:
        config (config_loader.ColocationConfig): Configuration

    Returns:
        Nothing
    """

    search_configs = []
    np.random.seed(config.seed)

    for _ in range(config.searching_count):
        new_config = deepcopy(config)
        new_config.selected_rooms = list(
            _choose_random_room(config.total_room_count, config.room_count))
        search_configs.append(new_config)

    sys.stderr.write('Loaded all configs\n')
    pool = multiprocessing.Pool(1)
    accuracies = list(pool.map(_eval_room_config, search_configs, chunksize=8))

    with open(
            config.base_file_name + 'search_result.csv', 'w',
            newline='') as file:
        header = [str(i) for i in range(config.room_count)]
        header.append('accuracy')
        csv_writer: csv.writer = csv.writer(file)
        for (conf, accuracy) in zip(search_configs, accuracies):
            csv_writer.writerow([*conf.selected_rooms, accuracy])

    return 0, 0, {}

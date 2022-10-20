"""Find patterns of rooms that lead to higher accuracy.
"""
import argparse
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

__ROOM_NAMES__ = ['413', '415', '417', '419', '421', '422', '423', '424',
                  '442', '446', '448', '452', '454', '456', '458', '462',
                  '510', '513', '552', '554', '556', '558', '562', '564',
                  '621', '621A', '621C', '621D', '621E', '640', '644', '648',
                  '656A', '656B', '664', '666', '668', '717', '719', '721',
                  '722', '723', '724', '725', '726', '734', '746', '748',
                  '752', '754', '776']


def _get_rooms_names()->list:
    return __ROOM_NAMES__


def single_room_distribution(rooms, accuracy: np.ndarray, base_path: pathlib.Path):
    """Distribution of accuracy regarding to single rooms
    """
    distribution = np.zeros((51, rooms.shape[0]), dtype=np.float)
    counts = np.zeros(51, dtype=np.int32)

    for row, accu in zip(rooms, accuracy):
        distribution[row, counts[row]] = accu
        counts[row] += 1

    trimmed = [distribution[i, :counts[i]] for i in range(len(distribution))]
    plt.figure(figsize=(12, 8))
    plt.boxplot(trimmed)
    plt.xticks(list(range(1, 52)), _get_rooms_names(),
               rotation=90, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(base_path.joinpath('single_room_distribution.png'), dpi=300)


def _plot_heat_map(data, path):
    plt.figure(figsize=(12, 8))
    plt.imshow(data, cmap='bwr')
    plt.tick_params(axis='x', labeltop=True, labelbottom=False,
                    top=True, bottom=False, labelrotation=90)
    plt.xticks(list(range(51)), _get_rooms_names())
    plt.yticks(list(range(51)), _get_rooms_names())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def double_room_distribution(rooms, accuracy: np.ndarray, base_path: pathlib.Path):
    """Distribution of accuracy regarding to pairs of rooms
    """
    distribution = np.zeros((51, 51, rooms.shape[0]), dtype=np.float)
    counts = np.zeros((51, 51), dtype=np.int32)

    for row, accu in zip(rooms, accuracy):
        for i in row:
            distribution[i, row, counts[i, row]] = accu
            counts[i, row] += 1

    means = np.sum(distribution, axis=2) / (counts + 0.0001)
    probability_of_1 = np.sum(distribution == 1.0, axis=2) / (counts + 0.0001)
    _plot_heat_map(means, base_path.joinpath('double_room_mean_accuracy.png'))
    _plot_heat_map(probability_of_1, base_path.joinpath('double_room_is1.png'))


def main():
    """Main entry point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('-o', '--output_dir', required=True)

    args = parser.parse_args(sys.argv[1:])
    base_path = pathlib.Path(args.output_dir)
    for filename in args.filenames:
        data = np.loadtxt(filename, dtype=float, delimiter=',')
        rooms = data[:, :-1].astype(int)
        accuracies = data[:, -1]

        output_path = base_path.joinpath(filename)
        output_path.mkdir(parents=True, exist_ok=True)

        single_room_distribution(rooms, accuracies, output_path)
        double_room_distribution(rooms, accuracies, output_path)


if __name__ == '__main__':
    main()

"""Generate and evaluate random solutions
"""
import numpy as np
from ..data_loader.config_loader import ColocationConfig
from ..data_loader.matrix_loader import load_matrix
from ..core import corr_score
from ..core import calcs


def run(config: ColocationConfig):
    """Evaluating random solutions

    Args:
        config (ColocationConfig): Configuration

    Returns:
        best fitness, accuracy of best solution, cached values
    """

    solution = np.arange(config.room_count * config.type_count)\
                 .reshape(config.room_count, config.type_count)

    cache: dict = {
        'fitnesses': [],
        'best_fitness': [],
        'best_solutions': [],
        'accuracies': []
    }

    corr_matrix = load_matrix(config.corr_matrix_path)
    corr_func = corr_score.compile_solution_func(corr_matrix,
                                                 config.type_count)

    for iteration in range(config.max_iteration):
        for t in range(1, config.type_count):
            np.random.shuffle(solution[:, t])

        cache['fitnesses'].append(corr_func(solution))
        cache['accuracies'].append(calcs.calculate_accuracy(solution))

        if config.verbose and iteration % 10 == 0:
            print('iteration[{}]'.format(iteration))

    cache['best_fitness'] = cache['fitnesses']

    return max(cache['fitnesses']), max(cache['accuracies']), cache

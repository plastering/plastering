"""Just return ground truth
"""
import numpy as np
from ..core import calcs
from ..core import corr_score
from ..data_loader import matrix_loader


def run(config):
    """Evaluate ground truth
    """
    solution = np.arange(config.room_count * config.type_count)
    solution = solution.reshape(config.room_count, config.type_count)

    corr_matrix = matrix_loader.load_matrix(config.corr_matrix_path)

    corr_func = corr_score.compile_solution_func(corr_matrix,
                                                 config.type_count)

    mean_score = corr_func(solution)

    accuracy = calcs.calculate_accuracy(solution)

    return mean_score, accuracy, {}

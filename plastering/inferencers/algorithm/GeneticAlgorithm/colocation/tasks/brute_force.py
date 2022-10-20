"""Brute force task
"""
import math
import numpy as np
from progressbar import ProgressBar

from ..core import calcs as C
from ..core import corr_score
from ..data_loader import matrix_loader
from ..data_loader.config_loader import ColocationConfig
from ..utils import printing


def run(config: ColocationConfig):
    """Run a bruteforce optimizer

    Args:
        config (ColocationConfig): configuration

    Returns:
        best score, best accuracy, a cache dict containing all accuracies
    """
    # Prepare a progress bar
    total_permutation = math.factorial(config.room_count)\
                        **(config.type_count - 1)
    progress_bar = ProgressBar(max_value=total_permutation)

    # Load the matrix
    corr_matrix = matrix_loader.load_matrix(config.corr_matrix_path)

    # If necessary, choose rooms
    if config.selected_rooms:
        corr_matrix = matrix_loader.select_rooms(
            corr_matrix, config.selected_rooms, config.type_count)
        assert corr_matrix.shape == (config.type_count * config.room_count,
                                     config.type_count * config.room_count)

    # Generate correlational coefficient calculator
    corr_func = corr_score.compile_solution_func(corr_matrix,
                                                 config.type_count)

    # Prepare the solution object
    solution = np.arange(config.room_count * config.type_count, dtype=np.int32)\
                 .reshape(config.room_count, config.type_count)

    # Record max correlational coefficients
    best_solutions = []  # Record all solutions achieving max corr coef
    best_corr_score = -np.inf

    # Fix the first sensor type, permute the rest
    for i, _ in enumerate(C.permute_axis_1(solution.T[1:])):
        # Evaluate the new solution
        score = corr_func(solution)
        if score > best_corr_score:
            best_corr_score = score
            best_solutions = [np.copy(solution)]
        elif score == best_corr_score:
            best_solutions.append(solution)

        if config.verbose:
            progress_bar.update(i)

    # Find accuracies of all saved solutions
    accuracies = [C.calculate_accuracy(s) for s in best_solutions]

    # Print result
    print(
        printing.as_table([['Corr. Coef.', best_corr_score],
                           ['Accuracy', max(accuracies)]]))

    # Print solution if required
    if config.print_final_solution:
        print(best_solutions[np.argmax(accuracies)])

    return best_corr_score, max(accuracies), {'accuracies': accuracies}

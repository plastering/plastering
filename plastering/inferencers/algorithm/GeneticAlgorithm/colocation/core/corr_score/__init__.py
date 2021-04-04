"""Some cython functions that speed up the computation
"""
import numpy as np
from . import c_score_func


def compile_room_func(corr_matrix, type_count):
    """Compile a room corr score function
    """
    mean_func = {
        4: c_score_func.room_mean_4,
    }.get(type_count, c_score_func.room_mean_generic)

    def wrapper(room):
        """Wrapper for a mean correlational score function.
        """
        return mean_func(corr_matrix, room.astype(np.int32))

    return wrapper


def compile_solution_func(corr_matrix, type_count):
    """Compile a room corr score function
    """
    mean_func = {
        4: c_score_func.solution_mean_4,
    }.get(type_count, c_score_func.solution_mean_generic)

    def wrapper(solution):
        """Wrapper for a mean correlational score function.
        """
        score = mean_func(corr_matrix, solution.astype(np.int32))
        return score

    return wrapper

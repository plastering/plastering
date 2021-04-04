"""Load correlation matrix
"""
from typing import List
import scipy.io
import numpy as np


def load_matrix(matrix_file: str):
    """Load correlational coefficient matrix, and apply absolute value
    """
    # pylint: disable=E1101
    mat = scipy.io.loadmat(matrix_file)['corr']
    #mat = np.fabs(mat)
    return mat


def select_rooms(matrix: np.ndarray, selected_rooms: List[int],
                 type_count: int):
    """Select rows and columns from matrix based on a list of rooms

    Args:
        matrix (np.ndarray): correlational coefficient matrix
        selected_rooms (List[int]): IDs of selected rooms
        type_count (int): Number of types in current matrix

    Returns:
        np.ndarray: a matrix of correlational coefficient
        Let ``sensor_count`` be ``type_count * len(selected_rooms)``,
        The dimension of the returned matrix is (sensor_count, sensor_count)
    """

    sensor_ids = np.array([[r_id * type_count + j for j in range(type_count)]
                           for r_id in selected_rooms]).reshape(-1)
    return matrix[sensor_ids][:, sensor_ids]


def select_types(matrix: np.ndarray, selected_types: List[int],
                 original_type_count: int, room_count: int):
    """Select rows and columns from matrix according to selected types

    Args:
        matrix (np.ndarray): correlational coefficient matrix
        selected_types (List[int]): types selected
        original_type_count (int): original number of types
        room_count (int): the number of rooms

    Returns:
        np.ndarray: a new coefficient matrix,
        with side length = ``len(selected_types) * room_count``
    """

    sensor_ids = np.array(
        [[r_id * original_type_count + t_id for t_id in selected_types]
         for r_id in range(room_count)]).reshape(-1)
    return matrix[sensor_ids][:sensor_ids]

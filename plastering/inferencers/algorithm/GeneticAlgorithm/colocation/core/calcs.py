"""Functional helpers that perform essential math/algorithmic calculations
"""

import itertools
import numpy as np


def _swap_element(arr, i, j):
    arr[i] ^= arr[j]
    arr[j] ^= arr[i]
    arr[i] ^= arr[j]


def permute(arr: np.array, num=-1):
    """permute an np array
    """
    if num == -1:
        num = arr.shape[0]

    if num == 1:
        yield arr
    else:
        for i in range(num - 1):
            for _ in permute(arr, num - 1):
                yield arr
            if num % 2 == 0:
                _swap_element(arr, i, num - 1)
            else:
                _swap_element(arr, 0, num - 1)
        for _ in permute(arr, num - 1):
            yield arr


def permute_axis_1(arr: np.array, level=-1):
    """Permute along axis 1
    """
    if level == -1:
        level = arr.shape[0]

    if level == 0:
        yield arr
    else:
        for _ in permute(arr[level - 1]):
            for _ in permute_axis_1(arr, level - 1):
                yield arr


def multi_level_permutation(lists, level):
    """Yielding through a multi-level permutatated list
    """
    if level == 0:
        yield []
    else:
        for perm in itertools.permutations(range(lists)):
            for rest in multi_level_permutation(lists, level - 1):
                yield [list(perm), *rest]


def all_pairs(arr):
    """Generate all possible pairs from an array
    """
    length = len(arr)
    for i in range(length - 1):
        for j in range(i + 1, length, 1):
            yield [arr[i], arr[j]]


def num_pairs(num_elements):
    """Calculate the number of pairs
    """
    return (num_elements * (num_elements - 1)) // 2


def arg_n_max(arr, num):
    """Get the indexes (unsorted) of the N largest elements
    """
    return np.argpartition(-arr, num)[:num]


def arg_n_min(arr, num):
    """Get the indexes (unsorted) of the N smallest elements
    """
    return np.argpartition(arr, num)[:num]


def sum_normalize(arr, axis=None):
    """Return the normalized array
    """
    arr = arr - np.min(arr)
    return arr / np.sum(arr, axis=axis, keepdims=True)


def calculate_accuracy(solution):
    """Calculate the accuracy of a solution
    """
    right_pairs = 0
    wrong_pairs = 0
    type_count = solution.shape[1]
    for room in solution:
        for (i, j) in all_pairs(room):
            if i // type_count == j // type_count:
                right_pairs += 1
            else:
                wrong_pairs += 1
    return right_pairs / (right_pairs + wrong_pairs)

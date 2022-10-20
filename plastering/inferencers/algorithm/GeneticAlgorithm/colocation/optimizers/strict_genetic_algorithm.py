"""A genetic algorithm that strictly ensure every individual is valid.
"""
import numpy as np
from ..core import calcs


def mutate_by_swap_weighted(population, child_id, mutation_rate, weights):
    """Perform mutation by randomly swapping pairs of rooms within a type.
    """
    solution = population[child_id]
    room_count, type_count = solution.shape
    pair_count = np.random.poisson(mutation_rate * room_count / 2, type_count)
    pair_count = np.clip(pair_count, 0, room_count / 2).astype(int)

    sequence = np.arange(0, room_count)

    for type_id, count in enumerate(pair_count):
        if count == 0:
            continue
        ids = np.random.choice(sequence, count * 2, replace=False, p=weights)
        swap_i, swap_j = ids[:count], ids[count:]
        solution[swap_i, type_id] ^= solution[swap_j, type_id]
        solution[swap_j, type_id] ^= solution[swap_i, type_id]
        solution[swap_i, type_id] ^= solution[swap_j, type_id]

    return solution


def crossing_over(population, child_id, father_id, crossing_over_rate):
    """Performing crossing over on two solutions, modifying the first one

    When $T$ types are present, manipulating $T - 1$ is sufficient.
    Therefore, type 0 is fixed.
    """
    child = population[child_id]
    father = population[father_id]
    type_count = child.shape[1]
    should_cross = np.random.rand(type_count - 1) < crossing_over_rate
    for type_id in range(1, type_count):
        if should_cross[type_id - 1]:
            child[:, type_id] = father[:, type_id]


def initialize_population(population_size, room_count, type_count):
    """Initialize a population
    """
    ind = np.arange(room_count * type_count, dtype=np.int32)\
            .reshape(1, room_count, type_count)
    population = np.repeat(ind, population_size, axis=0)
    for solution in population:
        for type_id in range(type_count):
            np.random.shuffle(solution[:, type_id])
    return population


def fitness(population, corr_score_func) -> np.ndarray:
    """calculate fitness scores of a population

    Args:
        population ([type]): [description]
        corr_score_func ([type]): [description]

    Returns:
        np.ndarray: [description]
    """

    fitnesses = np.zeros((population.shape[0], ), dtype=float)
    for p_id, individual in enumerate(population):
        fitnesses[p_id] = corr_score_func(individual)
    assert not np.isnan(fitnesses).any(), 'fitness has nan'
    return fitnesses


def suvival_of_fittest(fitnesses, survivor_count, replaced_count):
    """Generate the survived and lost id lists

    Returns: best ID, Winners' ID, Losers' ID
    """
    winner_ids = calcs.arg_n_max(fitnesses, survivor_count)
    loser_ids = calcs.arg_n_min(fitnesses, replaced_count)

    return np.argmax(fitnesses), winner_ids, loser_ids


def next_gen(population, survivor_ids, loser_ids, crossing_over_rate,
             mutation_rate, weight_func):
    """Generate the next generation (in place)
    """
    loser_count = len(loser_ids)
    mothers = np.random.choice(survivor_ids, size=loser_count)
    fathers = np.random.choice(survivor_ids, size=loser_count)

    for i, l_id in enumerate(loser_ids):
        m_id, f_id = mothers[i], fathers[i]
        population[l_id] = population[m_id]

        if f_id != m_id:
            crossing_over(population, l_id, f_id, crossing_over_rate)

        weights = None
        if weight_func is not None:
            # This is a performance critical area
            # Use for loop and np array instead of list-comp
            f_room = np.zeros(population.shape[1], dtype=np.float)
            for r_id, room in enumerate(population[l_id]):
                f_room[r_id] = weight_func(room)
            weights = calcs.sum_normalize(0 - f_room, axis=0)

        mutate_by_swap_weighted(population, l_id, mutation_rate, weights)

    return population

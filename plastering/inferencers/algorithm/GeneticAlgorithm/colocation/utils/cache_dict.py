"""Define cache format for storing history of training
"""

from typing import Dict


def get_cache(config) -> Dict[str, list]:
    """Prepare a cache dictionary

    Args:
        config (ColocationConfig): configuration file

    Returns:
        Dict[str, list]: cache dict
    """

    cache: Dict[str, list] = {}
    if config.plot_fitness_density:
        cache['fitnesses'] = []
    if config.plot_fitness_accuracy:
        cache['best_fitness'] = []
        cache['accuracies'] = []
    return cache

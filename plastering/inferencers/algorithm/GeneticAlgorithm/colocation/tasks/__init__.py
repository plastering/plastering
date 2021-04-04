"""Task to run
"""
from . import strict_ga
from . import random_solution
from . import ground_truth
from . import search_accurate_rooms
from . import room_combination_exhaustive
from . import brute_force
from . import consecutive_room

TASKS = {
    'strict_ga': strict_ga,
    'random_solution': random_solution,
    'ground_truth': ground_truth,
    'search_accurate_rooms': search_accurate_rooms,
    'room_combination_exhaustive': room_combination_exhaustive,
    'brute_force': brute_force,
    'consecutive_room': consecutive_room,
}

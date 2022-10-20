"""Main Module to run colocation optimizer
"""
import numpy as np
from . import tasks
from .utils import visualization
from .utils import paths
from .utils import arg_parser
from .utils import profiling


def ga(path_m, path_c):
    """Running colocation solver
    """
    #config = arg_parser.parse_args()
    config = arg_parser.my_parse_args(['-m', path_m, '-c', path_c])
    print(config)

    # Process the configuration
    paths.create_dir(config.base_file_name)
    profiler = profiling.Profiler(config)
    np.random.seed(config.seed)

    if config.task in tasks.TASKS:
        task = tasks.TASKS[config.task]
        profiler.start()
        best_solution, acc, cache, ground_truth_fitness, best_fitness = task.run(config)
        profiler.stop()

        if config.visualize:
            visualization.plot_cache(cache, config)

        profiler.print_results()
    else:
        print("unknown task name.")
    return best_solution, acc, ground_truth_fitness, best_fitness


if __name__ == '__main__':
    #main()
    pass

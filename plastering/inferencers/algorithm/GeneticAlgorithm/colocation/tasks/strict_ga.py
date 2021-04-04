"""Strict GA Task
"""
import numpy as np
from ..optimizers import strict_genetic_algorithm as ga
from ..core import calcs
from ..data_loader import matrix_loader
from ..data_loader import config_loader
from ..utils import cache_dict
from ..utils import printing
from ..core import corr_score

def ground_truth(individual):
    g_t = []
    for i in range(len(individual)):
        g_t.append([])
        for j in range(len(individual[i])):
            g_t[i].append(i*len(individual[i])+j)
    return g_t

def cal_acc(individual):
    pp = 0
    pn = 0
    for i in range(len(individual)):
        for j in range(len(individual[i]) - 1):
            for k in range(j + 1, len(individual[i])):
                if(int(individual[i][j] / 4) == int(individual[i][k] / 4)):
                    pp += 1
                else:
                    pn += 1
    recall = pp / (pp + pn)
    return recall

def run(config: config_loader.ColocationConfig):
    """run a strict Genetic optimizer

    Args:
        config (dict): configurations

    Returns:
        best fitness (float), accuracy of best solution, cache
    """
    # Prepare cache
    cache = cache_dict.get_cache(config)

    # Prepare verbose print function
    vprint = printing.compile_vprint_function(config.verbose)

    # Load matrix
    corr_matrix = matrix_loader.load_matrix(config.corr_matrix_path)

    # If necessary, choose rooms
    if config.selected_rooms:
        corr_matrix = matrix_loader.select_rooms(corr_matrix, config.selected_rooms,
                                                 config.type_count)
        assert corr_matrix.shape == (config.type_count * config.room_count,
                                     config.type_count * config.room_count), str(
                                         config.type_count) + ' ' + str(config.room_count)

    # Compile functions
    corr_func = corr_score.compile_solution_func(corr_matrix, config.type_count)

    weight_func = corr_score.compile_room_func(
        corr_matrix, config.type_count) if config.mutation_weighted else None

    population = ga.initialize_population(config.population_count, config.room_count,
                                          config.type_count)
    best_fitness = 0
    best_solution = None

    assert not np.isnan(population).any(), 'population has nan'
    for iteration in range(config.max_iteration):
        fitnesses: np.ndarray = ga.fitness(population, corr_func)

        best, winners, losers = ga.suvival_of_fittest(fitnesses, config.survivor_count,
                                                      config.replaced_count)

        best_fitness = fitnesses[best]
        best_solution = np.copy(population[best])

        if iteration % 100 == 0 and config.verbose:
            vprint('Iteration [{}]: {}', iteration, fitnesses[best])
            '''
            recalls = []
            for i in range(len(population)):
                recalls.append(cal_acc(population[i]))
            recalls = np.array(recalls)

            ids = np.argsort(fitnesses)
            for id in ids:
                print(id, " Fitness %f; Recall %f" % (fitnesses[id], recalls[id]))
            '''
        if config.plot_fitness_density:
            cache['fitnesses'].append(fitnesses)
        if config.plot_fitness_accuracy:
            cache['best_fitness'].append(fitnesses[best])
            cache['accuracies'].append(calcs.calculate_accuracy(population[best]))
        
        population = ga.next_gen(
            population,
            winners,
            losers,
            config.crossing_over_rate,
            config.mutation_rate,
            weight_func=weight_func)

    g_t = ground_truth(population[0])
    fit: np.ndarray = ga.fitness(np.array([g_t], dtype = np.int32), corr_func)

    recalls = []
    for i in range(len(population)):
        recalls.append(cal_acc(population[i]))
    recalls = np.array(recalls)

    ids = np.argsort(fitnesses)
    for id in ids:
        print("Fitness %f; Recall %f" % (fitnesses[id], recalls[id]))
    

    print("Ground Truth fitness:", fit[0])
    if config.print_final_solution:
        print('Final Solution:')
        print(best_solution)

    return best_solution, calcs.calculate_accuracy(best_solution), cache, fit[0], best_fitness

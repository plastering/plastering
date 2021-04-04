"""Calculate correlational coefficient efficiently
"""

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
cimport cython


############################# Pure C Functions #############################

@cython.boundscheck(False)
cdef double c_room_mean_4(double[:, :] corr_matrix, int[:] room) nogil:
    """Fast calculating room's mean
    """
    return (corr_matrix[room[0], room[1]]
            + corr_matrix[room[0], room[2]]
            + corr_matrix[room[0], room[3]]
            + corr_matrix[room[1], room[2]]
            + corr_matrix[room[1], room[3]]
            + corr_matrix[room[2], room[3]]
            ) / 6.0


@cython.boundscheck(False)
cdef double c_room_mean_generic(double[:, :] corr_matrix, int[:] room) nogil:
    """Fast calculating room's mean"""
    cdef double sum = 0
    cdef int count = 1
    cdef int type_count = room.shape[0]
    cdef int i, j, room_i, room_j
    for i in range(type_count - 1):
        room_i = room[i]
        for j in range(i + 1, type_count):
            room_j = room[j]
            count += 1
            sum += corr_matrix[room_i, room_j]

    return sum / count


@cython.boundscheck(False)
cdef double c_solution_mean_4(double[:, :] corr_matrix, int[:, :] solution) nogil:
    """calculate the mean correlation of a solution
    """
    cdef int i, room_count = solution.shape[0]
    cdef double sum = 0
    cdef double* results = <double*> malloc(sizeof(double) * room_count)
    with nogil, parallel():
        for i in prange(room_count, schedule='guided'):
            results[i] = (corr_matrix[solution[i][0], solution[i][1]]
                            + corr_matrix[solution[i][0], solution[i][2]]
                            + corr_matrix[solution[i][0], solution[i][3]]
                            + corr_matrix[solution[i][1], solution[i][2]]
                            + corr_matrix[solution[i][1], solution[i][3]]
                            + corr_matrix[solution[i][2], solution[i][3]]
                            ) / 6.0
    
    for i in range(room_count):
        sum += results[i]
    
    free(results)
    return sum / room_count


@cython.boundscheck(False)
cdef double c_solution_mean_generic(double[:, :] corr_matrix, int[:, :] solution) nogil:
    """calculate the mean correlation of a solution
    """
    cdef int i, room_count = solution.shape[0]
    cdef double sum = 0
    for i in range(room_count):
        sum += c_room_mean_generic(corr_matrix, solution[i])
    return sum / room_count


########################### Wrappers for python ##########################


cpdef room_mean_4(double[:, :] corr_matrix, int[:] room):
    """Calculate the mean of correlational coefficients of 4 sensors
    """
    return c_room_mean_4(corr_matrix, room)

cpdef room_mean_generic(double[:, :] corr_matrix, int[:] room):
    """Calculate the mean of correlational coefficient for any number of sensors
    """
    return c_room_mean_generic(corr_matrix, room)

cpdef solution_mean_4(double[:, :] corr_matrix, int[:, :] solution):
    """Calculate the mean of correlational coefficient for a solution where type
    count is 4.
    """
    score = c_solution_mean_4(corr_matrix, solution)
    return score

cpdef solution_mean_generic(double[:, :] corr_matrix, int[:, :] solution):
    """Calculate the mean of correlational coefficient for a solution
    """
    return c_solution_mean_generic(corr_matrix, solution)



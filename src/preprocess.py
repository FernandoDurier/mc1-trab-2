import numpy as np

def clean(dataset):
    return np.matrix(dataset[1:])

def select(mean_vec, size):
    if mean_vec.ndim != 1:
        raise ValueError('Mean vector parameter must be a 1d array')
    
    pivot = mean_vec.size - size
    lower_bound = np.partition(mean_vec, pivot)[-size]
    mean_list = list(mean_vec)
    return [index + 1 for index, mean in enumerate(mean_list) if mean >= lower_bound]


def n_method(matrix, size):
    matrix = matrix[:, 0:42].astype(np.float64)

    # axis=0 is mean of columns 
    # A1 attribute is flattened 1d array
    means = np.matrix.mean(matrix, axis=0).A1
    return select(means, size)

def p_method():
    pass

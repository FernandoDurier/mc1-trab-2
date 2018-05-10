import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def clean(dataset):
    matrix = np.matrix(dataset[1:])
    matrix[matrix==''] = '-1'
    return matrix

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

def p_method(matrix, size):
    user_attrs = matrix[:, 42:].astype(np.float64)
    features = PCA(n_components=2).fit_transform(user_attrs)
    clusters = KMeans(n_clusters=2).fit_predict(user_attrs)

    matrix = matrix[:, 0:42].astype(np.float64)
    cluster_1 = matrix[clusters == 1]
    cluster_0 = matrix[clusters == 0]

    means_1 = np.matrix.mean(cluster_1, axis=0).A1
    means_0 = np.matrix.mean(cluster_0, axis=0).A1

    return select(means_0, size) + select(means_1, size)

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        cluster_indices, clusters = get_clusters_coordinates(X, y)
        a = calculate_a(cluster_indices, clusters)
        b = calculate_b(cluster_indices, clusters)
        return np.divide(np.subtract(b, a), np.maximum(b, a))


def get_clusters_coordinates(X: np.ndarray, y: np.ndarray) -> Tuple[List, List]:
    """
    Sorts unordered cluster coordinates

    inputs:
        X: np.ndarray
            A 2D matrix where the rows are observations and columns are features.

        y: np.ndarray
            a 1D array representing the cluster labels for each of the observations in `X`

    outputs:
        list of indices from the original unsorted order of coordinates, sorted by cluster
        list of features sorted by cluster
    """
    all_clusters_coordinates = []
    all_cluster_indices = []
    num_clusters = np.max(y) + 1
    for cluster_index in range(num_clusters):
        cluster_indices = np.where(y == cluster_index)[0]
        cluster_coordinates = X[cluster_indices][:]
        all_clusters_coordinates.append(cluster_coordinates)
        all_cluster_indices.append(cluster_indices)
    return all_cluster_indices, all_clusters_coordinates


def calculate_a(indices: List, clusters: List) -> np.ndarray:
    """
    calculates how far each point in the cluster is from other points in the same cluster

    inputs:
        indices: List
            Indices from the original unsorted order of coordinates, sorted by cluster

        cluster: List
            features sorted by cluster

    outputs:
        np.ndarray
            a 1D array with the silhouette "a" values for each observation
    """
    a = np.empty(np.hstack(indices).size)

    for cluster_label, cluster in enumerate(clusters):
        distances_matrix = cdist(cluster, cluster)

        # mean distance to every other dot: divide sum by n-1 because distance to self is zero
        a_this_cluster = np.sum(distances_matrix, axis=0) / (distances_matrix.shape[0] - 1)
        np.put(a, indices[cluster_label], a_this_cluster)
    return a


def calculate_b(indices, clusters) -> np.ndarray:
    """
        calculates the smallest mean distance for each point in a cluster to pints in a different
        cluster

        inputs:
            indices: List
                Indices from the original unsorted order of coordinates, sorted by cluster

            cluster: List
                features sorted by cluster

        outputs:
            np.ndarray
                a 1D array with the silhouette "b" values for each observation
        """
    b = np.zeros(np.hstack(indices).size)
    num_clusters = len(clusters)
    if num_clusters < 2:
        raise ValueError("There need to be at least 2 clusters to run silhouette score. There is "
                         "only 1 cluster.")
    for cluster_label, cluster in enumerate(clusters):
        other_cluster_labels = list(range(num_clusters))
        other_cluster_labels.remove(cluster_label)
        first_other_cluster = clusters[other_cluster_labels[0]]
        b_this_cluster = np.mean(cdist(cluster, first_other_cluster), axis=1)
        other_cluster_labels.pop(0)
        for other_cluster_label in other_cluster_labels:
            other_cluster = clusters[other_cluster_label]
            mean_dist_other_cluster = np.mean(cdist(cluster, other_cluster), axis=1)
            b_this_cluster = np.minimum(b_this_cluster, mean_dist_other_cluster)
        np.put(b, indices[cluster_label], b_this_cluster)
    return b

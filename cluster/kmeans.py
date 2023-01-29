import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        if not k > 0:
            raise ValueError("k needs to be at least 1")
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        cluster_center_values = self._define_random_centroids(mat)

        for _ in range(self.max_iter):
            assigned_clusters = assign_cluster(cluster_center_values, mat)
            new_cluster_center_values = self._calculate_new_kmean(assigned_clusters, mat)
            mse = mean_squared_error(cluster_center_values, new_cluster_center_values)
            if mse < self.tol:
                break
            else:
                cluster_center_values = new_cluster_center_values

        self.cluster_centers = new_cluster_center_values
        self.mse = mse

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        _, num_features_fit = self.cluster_centers.shape
        _, num_features_predict = mat.shape
        if num_features_fit != num_features_predict:
            raise ValueError(f"Fit data and predict data need to have the same number of "
                             f"dimensions or features. Fit data has {num_features_fit} dimensions "
                             f"while predict data has {num_features_predict}.")

        return assign_cluster(self.cluster_centers, mat)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.mse

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.cluster_centers

    def _define_random_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        pick random observations

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                features of some random k (number of clusters) observations in mat
        """
        num_observations, num_features = mat.shape
        cluster_center_indices = []
        for _ in range(self.k):
            random_observation_index = random.randint(0, num_observations - 1)
            while random_observation_index in cluster_center_indices:
                random_observation_index = random.randint(0, num_observations - 1)
            cluster_center_indices.append(random_observation_index)
        cluster_center_values = mat[cluster_center_indices][:]
        return cluster_center_values

    def _calculate_new_kmean(self, past_clusters: np.ndarray, mat: np.ndarray):
        """
        calculate new centroid of each previously-defined cluster

        inputs:
            past_clusters: np.ndarray
                A 1D array with the cluster label for each of the observations in `mat`
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                A 2D matrix of features of centroids calculated for each cluster
        """
        kmeans_new = []
        for cluster_index in range(self.k):
            cluster_indices = np.where(past_clusters == cluster_index)[0]
            cluster_coordinates = mat[cluster_indices][:]
            new_kmean_coordinates = np.mean(cluster_coordinates, axis=0)
            kmeans_new.append(new_kmean_coordinates)
        return np.array(kmeans_new)


def assign_cluster(cluster_coordinates: np.ndarray, mat: np.ndarray):
    """
    Assigns observations to clusters based on the cluster centroids

    inputs:
        cluster_coordinates: np.ndarray
            A 2D matrix of features of centroids calculated for each cluster
        mat: np.ndarray
            A 2D matrix where the rows are observations and columns are features

    outputs:
        np.ndarray
            a 1D array with the cluster label for each of the observations in `mat`
    """
    distances = cdist(cluster_coordinates, mat)
    assigned_clusters = np.argmin(distances, axis=0)
    return assigned_clusters

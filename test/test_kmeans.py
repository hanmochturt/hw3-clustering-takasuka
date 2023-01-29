import pytest
import random
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from cluster import KMeans, make_clusters


def test_kmeans_fit_predict():
    """
    unit test for kmeans fit and predict functions
    By definition, the cluster centers should predict to identify with their corresponding clusters.
    If one of each cluster center is predicted, then the resulting numpy array should be equal to
    the range of the number of clusters
    """
    num_clusters = random.randint(1, 10)
    clusters, labels = make_clusters(k=num_clusters, scale=1)
    km = KMeans(k=num_clusters)
    km.fit(clusters)
    pred = list(km.predict(km.get_centroids()))
    pred.sort()
    assert pred == list(range(num_clusters))


def test_kmeans_low_k_error():
    """
    unit test for initializing kmeans function
    should raise an error for 0 clusters because there needs to be at least 1 cluster for the
    function to work
    """
    with pytest.raises(ValueError):
        KMeans(k=0)


def test_kmeans_fit_predict_num_features_mismatch_error():
    """
    unit test for fitting data with a different number of features than the predict data
    """
    num_clusters = random.randint(1, 10)
    num_features_fit = random.randint(1, 10)
    num_features_predict = random.randint(1, 10)
    while num_features_fit == num_features_predict:
        num_features_predict = random.randint(1, 10)
    clusters, labels = make_clusters(k=num_clusters, m=num_features_fit, scale=1)
    km = KMeans(k=num_clusters)
    km.fit(clusters)
    with pytest.raises(ValueError):
        km.predict(make_clusters(m=num_features_predict)[1])

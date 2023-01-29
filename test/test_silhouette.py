import pytest
import numpy as np
import random
import sklearn
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from cluster import KMeans, Silhouette, make_clusters


def test_silhouette_score():
    """
    unit test for comparing the mean silhouette score calculated by my silhouette class vs.
    sklearn's method
    """
    num_clusters = random.randint(2, 10)
    clusters, labels = make_clusters(k=num_clusters, scale=1)
    km = KMeans(k=num_clusters)
    km.fit(clusters)
    pred = km.predict(clusters)
    my_score = np.mean(Silhouette().score(clusters, pred))
    sklearn_score = sklearn.metrics.silhouette_score(clusters, pred)
    num_decimals_my_score = len(str(my_score)) - 2
    num_decimals_sklearn_score = len(str(sklearn_score)) - 2

    # account for small rounding errors
    if num_decimals_my_score != num_decimals_sklearn_score:
        if num_decimals_my_score < num_decimals_sklearn_score:
            sklearn_score = round(sklearn_score, num_decimals_my_score - 1)
            my_score = round(my_score, num_decimals_my_score - 1)
        else:
            my_score = round(my_score, num_decimals_sklearn_score - 1)
            sklearn_score = round(sklearn_score, num_decimals_sklearn_score - 1)
    assert my_score == sklearn_score


def test_silhouette_low_cluster_error():
    """
    unit test for attempting to calculate silhouette score with only 1 cluster
    """
    clusters, labels = make_clusters(k=1, scale=1)
    km = KMeans(k=1)
    km.fit(clusters)
    pred = km.predict(clusters)
    with pytest.raises(ValueError):
        np.mean(Silhouette().score(clusters, pred))

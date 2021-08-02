"""
test for iris dataset adding robust sample
"""
import faiss
from sklearn.datasets import load_iris
import numpy as np
from clustering.cluster import *
from utils.accuracy import accuracy, accuracy_iris, accuracy_noise
import random
from utils.add_noise import add_noise


if __name__ == '__main__':
    iris = load_iris()
    X = iris["data"]
    Y = iris["target"]

    X = preprocess(X)
    kmeans, images_lists, clus_index2dis, loss = run_kmeans(X, nmb_clusters=3, verbose=True)
    path2label = Y.tolist()
    accuracy_iris(images_lists, path2label)

    centroids = faiss.vector_to_array(kmeans.centroids).reshape(3, X.shape[1])
    X_noise, path2label2 = add_noise(X, path2label, centroids)

    X_noise = preprocess(X_noise)

    k2, images_lists2, clus_index2dis2, loss2 = run_kmeans(X_noise, nmb_clusters=3, verbose=True)
    print('with noise: ', end=' ')
    accuracy_noise(images_lists2, path2label2)

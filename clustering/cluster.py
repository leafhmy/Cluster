import faiss
import numpy as np


np.random.seed(1234)

def compute_dis(X, clu_index, path_index, centroids):
    feature = X[path_index]
    clu_centroid = centroids[clu_index]
    dis = np.linalg.norm(feature - clu_centroid)
    return dis


def preprocess(npdata):
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    return npdata


def PCA(args, npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False, use_gpu=False, dist='L2'):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    if use_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        if dist == 'L2':
            index = faiss.GpuIndexFlatL2(res, d, flat_config)
        elif dist == 'IP':
            # 内积搜索
            # IP stands for "inner product".
            # If you have normalized vectors, the inner product becomes cosine similarity.
            index = faiss.GpuIndexFlatIP(res, d, flat_config)
    else:
        if dist == 'L2':
            index = faiss.IndexFlatL2(d)
        elif dist == 'IP':
            index = faiss.IndexFlatIP(d)

    # perform the training
    clus.train(x, index)
    #     return clus
    _, I = index.search(x, 1)

    # compute distance
    distance = []
    centroids = faiss.vector_to_array(clus.centroids)
    centroids = centroids.reshape(nmb_clusters, x.shape[1])
    images_lists = [[] for i in range(nmb_clusters)]
    for i in range(x.shape[0]):
        images_lists[I[i][0]].append(i)

    for clu_index, clu in enumerate(images_lists):
        for index in clu:
            distance.append(compute_dis(x, clu_index, index, centroids[clu_index]))

    clus_index2dis = [(clus_index, dis) for clus_index, dis in zip(I, distance)]

    # losses = faiss.vector_to_array(clus.obj)  # this option was replaced. The fix is:
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return clus, images_lists, clus_index2dis, losses[-1]
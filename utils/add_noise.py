import numpy as np
import random


# noise label must be -1
def add_noise(x, path2label, centroids, ratio=0.1):
    noise_num = int(len(path2label) * ratio)
    mean = []
    cls_num = centroids.shape[0]
    for i in range(0, cls_num-1):
        for j in range(i+1, cls_num):
            aux = np.vstack((centroids[i], centroids[j]))
            mean.append(np.mean(aux, axis=0))
    for i in range(noise_num):
        index = random.randint(0, len(mean)-1)
        mean_feature = mean[index]
        normal_mean = 0
        std = np.min(mean_feature)
        noise = mean_feature + np.random.normal(normal_mean, std, size=mean_feature.size)

        x = np.vstack((x, noise))
        path2label.append(-1)

    return x, path2label

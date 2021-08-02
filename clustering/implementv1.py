"""
原始图像PCA聚类
acc: 0.25609756097560976
"""
from clustering.cluster import *
import numpy as np
from collections import Counter


def accuracy(clu2index, path2label):
    aux = [[] for i in range(len(clu2index))]
    for i, clu in enumerate(clu2index):
        for x_index in clu:
            for index, (_, label) in enumerate(path2label):
                if x_index == index:
                    aux[i].append(label)

    # clu_index2label = {k: 0 for k in range(len(clu2index))}
    clu_index2label = np.empty(shape=(len(clu2index), len(clu2index)))
    axis2times = []
    for i, clu in enumerate(aux):
        c = Counter(clu)
        for k, v in c.items():
            clu_index2label[i, k] = v  # 第i个类簇k出现了v次
            axis2times.append(((i, k), v))

    axis2times.sort(key=lambda v: v[1], reverse=True)
    clu2label = {k: 0 for k in range(len(clu2index))}
    flag1 = {k: False for k in range(len(clu2index))}
    flag2 = {k: False for k in range(len(clu2index))}
    for t in axis2times:
        if t[1] > clu2label[t[0][0]]:
            if flag1[t[0][1]] or flag2[t[0][0]]:
                continue
            clu2label[t[0][0]] = t[0][1]
            flag1[t[0][1]] = True
            flag2[t[0][0]] = True

    correct = 0
    for i, clu in enumerate(clu2index):
        for index in clu:
            if path2label[index][1] == clu2label[i]:
                correct += 1

    acc = correct / len(path2label)
    print(f"acc: {acc}")


def ori_image_cluster(args, loader):
    train_loader = loader.get_data_loader()
    data = np.zeros(shape=(1, 3 * 224 * 224))
    for img, _ in train_loader:
        img = img.reshape(img.shape[0], -1).detach().cpu().numpy()
        data = np.vstack((data, img))

    data = data[1:]
    data = PCA(args, data)

    I, loss = run_kmeans(data, args.nmb_cluster, verbose=args.verbose, use_gpu=args.faiss_gpu)
    images_lists = [[] for i in range(args.nmb_cluster)]
    for i in range(len(data)):
        images_lists[I[i]].append(i)

    path2label = loader.path2label

    accuracy(images_lists, path2label)

    # return images_lists, path2label
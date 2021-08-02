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
        if all(flag1.values()) and all(flag2.values()):
            break
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



def accuracy_iris(clu2index, path2label):
    aux = [[] for i in range(len(clu2index))]
    for i, clu in enumerate(clu2index):
        for x_index in clu:
            # for index, (_, label) in enumerate(path2label):
            for index, label in enumerate(path2label):  # for iris
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
        if all(flag1.values()) and all(flag2.values()):
            break
        if t[1] > clu2label[t[0][0]]:
            if flag1[t[0][1]] or flag2[t[0][0]]:
                continue
            clu2label[t[0][0]] = t[0][1]
            flag1[t[0][1]] = True
            flag2[t[0][0]] = True

    correct = 0
    for i, clu in enumerate(clu2index):
        for index in clu:
            # if path2label[index][1] == clu2label[i]:
            if path2label[index] == clu2label[i]:  # for iris
                correct += 1

    acc = correct / len(path2label)
    print(f"acc: {acc}")


def accuracy_noise(clu2index, path2label):
    aux = [[] for i in range(len(clu2index))]
    for i, clu in enumerate(clu2index):
        for x_index in clu:
            # for index, (_, label) in enumerate(path2label):
            for index, label in enumerate(path2label):  # for iris
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
        if all(flag1.values()) and all(flag2.values()):
            break
        if t[1] > clu2label[t[0][0]]:
            if flag1[t[0][1]] or flag2[t[0][0]] or t[0][1] == -1:
                continue
            clu2label[t[0][0]] = t[0][1]
            flag1[t[0][1]] = True
            flag2[t[0][0]] = True

    correct = 0
    noise_num = 0
    for i, clu in enumerate(clu2index):
        for index in clu:
            # if path2label[index][1] == clu2label[i]:
            if path2label[index] == -1:
                noise_num += 1
                continue
            if path2label[index] == clu2label[i]:  # for iris
                correct += 1

    acc = correct / (len(path2label) - noise_num)
    print(f"acc: {acc}")





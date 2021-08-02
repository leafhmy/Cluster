import os
import time
import numpy as np
from PIL import Image
import os


class Loader:
    def __init__(self, root_dir):
        self.root = root_dir
        self.anno = './anno/'
        if not os.path.exists(self.anno):
            os.mkdir(self.anno)
        self._save_file_path()


    def _save_file_path(self):
        fpath = []
        labels = []

        for index, d in enumerate(os.listdir(self.root)):
            fd = os.path.join(self.root, d)
            for i in os.listdir(fd):
                fp = os.path.join(fd, i)
                fpath.append(fp)
                labels.append(index)


        with open(self.anno + 'train.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    def getDataLoader(self, train_trans):
        self.train_loader = DatasetLoader(self.anno, 'train', train_trans)
        return self.train_loader

    def get_path2label(self):
        assert self.train_loader is not None
        return self.train_loader.path2label


class DatasetLoader:
    def __init__(self, anno_dir, phase, data_transforms):
        self.path2label = []
        self.anno = anno_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.anno+phase+'.txt', 'r') as f:
            for item in f.readlines():
                img, label = item.split(' ')
                label = int(label.strip())
                self.data.append((img, label))
                self.path2label.append(label)

    def __len__(self):
        return len(self.data)

    def get_path2label(self):
        return self.path2label

    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = Image.open(img_path).convert('RGB')
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label


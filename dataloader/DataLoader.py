import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class WBCDataset:
    def __init__(self, args):
        self.args = args
        self.anno = './anno/'
        if not os.path.exists(self.anno):
            os.mkdir(self.anno)
        self._create_anno_v1()
        self.data_transforms = self.get_trans()
        self._create_loader()
        self.path2label = self._Loader.get_path2label()


    def get_trans(self, trans=None):
        if trans:
            return trans
        trans = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
        return trans

    def _create_anno_v1(self):
        fpath = []
        labels = []

        for index, d in enumerate(os.listdir(self.args.data)):
            fd = os.path.join(self.args.data, d)
            label = index
            for i in os.listdir(fd):
                fp = os.path.join(fd, i)
                fpath.append(fp)
                labels.append(label)

        with open(self.anno + 'train.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    def _create_loader(self):
        self._Loader = DatasetLoader(self.anno, 'train', self.data_transforms)
        self._loader = DataLoaderX(self._Loader,
                                  batch_size=self.args.batch,
                                  num_workers=self.args.workers,
                                  shuffle=False)

    def get_data_loader(self):
        return self._loader

    def get_path2label(self):
        return self.path2label


class DatasetLoader:
    def __init__(self, anno_dir, phase, data_transforms):
        self.anno = anno_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.anno+phase+'.txt', 'r') as f:
            for item in f.readlines():
                img, label = item.split(' ')
                self.data.append((img, int(label.strip())))

    def __len__(self):
        return len(self.data)

    def get_path2label(self):
        return self.data

    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = Image.open(img_path).convert('RGB')
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label

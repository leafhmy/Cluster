"""
DeepCluter / SimCLR模型聚类
"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import deepcluster_models as models
from clustering.cluster import *
from utils.accuracy import accuracy, accuracy_iris
from loaddata.dataloader import Loader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def getParser():
    parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                     of frozen convolutional layers of an AlexNet.""")

    parser.add_argument('--data', type=str, help='path to dataset',
                        # default='/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/')
                        default='E:/dataset/segmentation_datasets/images/')
    parser.add_argument('--model', type=str, help='path to model',
                        # default='D:\PythonProjects\deepcluster_checkpoint/checkpoint_0_alexnet.pth.tar')
                        default='D:\PythonProjects\deepcluster_checkpoint/checkpoint_0.pth_vgg16.tar')
    parser.add_argument('--tencrops', action='store_true',
                        help='validation accuracy averaged over 10 crops')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', type=int, default=31, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--f_dim', type=int, default=4096)
    args = parser.parse_args()
    return args


def get_features(model, loader, f_dim=4096):
    model.eval()
    feature = np.zeros(shape=(1, f_dim))
    for i, (input, _) in enumerate(loader):
        input = input.cuda()
        fea = model(input)
        fea = fea.detach().cpu().numpy()
        feature = np.vstack((feature, fea))
    return feature[1:]


def load_model(args):
    """Loads model and return it without DataParallel table."""
    path = args.model
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))
        model.top_layer = None

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


def load_model_simclr(args):
    path = args.model
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        model = torchvision.models.resnet18(pretrained=False, num_classes=args.f_dim).cuda()
        checkpoint = torch.load(path, map_location='cuda')
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model



def validate(args):
    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    # use DeepCLuster or SimCLR model
    # model = load_model_simclr(args)
    model = load_model(args)

    model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transformations_train = transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize])


    loader = Loader(args.data)
    train_dataset = loader.getDataLoader(train_trans=transformations_train)

    loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    path2label = train_dataset.path2label

    feature = get_features(model, loader, f_dim=args.f_dim)
    feature = preprocess(feature)
    # feature = PCA(args, feature, pca=256)

    clus, images_lists, clus_index2dis, loss = run_kmeans(feature, 5, verbose=True)
    accuracy_iris(images_lists, path2label)


if __name__ == '__main__':
    args = getParser()
    validate(args)



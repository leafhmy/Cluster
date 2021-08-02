from dataloader.DataLoader import WBCDataset
from clustering.implementv1 import *
from clustering.implementv2 import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='DIR', help='path to dataset',
                        default='E:\dataset\segmentation_datasets\WBC_images/')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=5,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--batch', default=8, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 31)')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='use faiss-gpu')
    parser.add_argument('--model', type=str, default='')
    return parser.parse_args()


def getParser_dc():
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
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='use faiss-gpu')
    args = parser.parse_args()
    return args


def main(args):
    loader = WBCDataset(args)
    ori_image_cluster(args, loader)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 13, 14'
    # args = parse_args()
    # main(args)
    args = getParser_dc()
    validate(args)

"""
python main.py --data '/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/' \
--model "/home/st003/hmy/DeepClusterSimCLRv3/checkpoint_dc_final_epoch201.pth.tar" \
--verbose --faiss_gpu
"""


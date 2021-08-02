from clustering.implementv2 import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 15'


if __name__ == '__main__':
    args = getParser()
    main(args)

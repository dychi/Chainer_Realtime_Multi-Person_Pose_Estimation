# coding: utf-8
import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.serializers import npz


chainer.config.train = False

class MyVGG(chainer.Chain):
    def __init__(self):
        super(MyVGG, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()

    def __call__(self, x):
        h = self.base(x, layers=['fc7'])['fc7']

        return h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpatialCNN')
    parser.add_argument('--imgs_dir')
    parser.add_argument('--list_file')
    args = parser.parse_args()

    model = MyVGG()
    model.base.disable_update()

    img_list_file = pd.read_csv(args.list_file, sep=',', usecols=[1,2])
    imgs_name = img_list_file["Img_name"]

    num = 2
    feat_array = []
    for i, img_path in enumerate(imgs_name):
        if i == 2:
            break
        img = Image.open(args.imgs_dir + '/{}'.format(img_path))
        x = L.model.vision.vgg.prepare(img)
        x = x[np.newaxis]
        result = model(x)
        feat_array.append(result[0])

    print(feat_array)
    print(np.array(feat_array).shape)

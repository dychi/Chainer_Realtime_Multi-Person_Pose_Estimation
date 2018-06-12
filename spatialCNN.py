# coding: utf-8
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links import caffe
from chainer.serializers import npz


model_npz = np.load('./VGG_CNN_M_2048.npz')

class BaseVGG(chainer.Chain):
    def __init__(self):
        super(BaseVGG, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=96, ksize=7, stride=2)
            self.conv2 = L.Convolution2D(96, out_channels=256, ksize=5, stride=2)
            self.conv3 = L.Convolution2D(256, out_channels=512, ksize=3, pad=1)
            self.conv4 = L.Convolution2D(512, out_channels=512, ksize=3, pad=1)
            self.conv5 = L.Convolution2D(512, out_channels=512, ksize=3, stride=2)
            
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(h, n=5, k=2, alpha=0.0005, beta=0.75)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        
        h = F.relu(self.conv2(h))
        h = F.local_response_normalization(h, n=5, k=2, alpha=0.0005, beta=0.75)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        
        h = F.relu(self.conv3(h))
        
        h = F.relu(self.conv4(h))
        
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        
        return h

    
class VGG(chainer.Chain):
    def __init__(self, pretrained_model=model_npz):
        super(VGG, self).__init__()
        with self.init_scope():
            self.base = BaseVGG()
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 2048)
    
    def __call__(self, x):
        h = self.predict(x)
        return h
    
    def predict(self, x):
        h = self.base(x)
        h = F.dropout(F.relu(self.fc6(h)), ratio=.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=.5)
        return h



def change_image_size(image_path):
    # 入力画像サイズの定義
    image_shape = (224, 224)
    # 画像を読み込み、RGB形式に変換する
    image = Image.open(image_path).convert('RGB')
    # 画像のリサイズとクリップ
    image_w, image_h = image_shape
    w, h = image.size
    if w > h:
        shape = [image_w * w / h, image_h]
    else:
        shape = [image_w, image_h * h / w]
    x = (shape[0] - image_w) / 2
    y = (shape[1] - image_h) / 2
    image = image.resize(image_shape)
    pixels = np.asarray(image).astype(np.float32)
    
    # pixels は3次元でそれぞれの軸は[Y座標, X座標, RGB]を表しているが、
    # 入力画像は4次元で[画像インデックス, BGR, Y座標, X座標]なので、配列の変換を行う
    # RGBからBGRに変換する
    pixels = pixels[:,:,::-1]
    # 軸を入れ替える
    pixels = pixels.transpose(2,0,1)
    # 4次元にする
    pixels = pixels.reshape((1,) + pixels.shape)
    return pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spatial CNN')
    parser.add_argument('--imgs_dir', help='original images directory')
    parser.add_argument('--img_list_file', help='path to text file which has image list') 
    args = parser.parse_args()
    
    # モデルの読み込み
    model = VGG(pretrained_model=model_npz)
    
    # 画像ディレクトリのpath
    img_list_file = pd.read_csv(args.img_list_file, sep=',', usecols=[1,2])
    imgs_name = img_list_file["Img_name"]
    
    feat_array = []
    for i, img_path in enumerate(imgs_name):
        if i == 5:
            break
            
        pixs = change_image_size(args.imgs_dir + '/{}'.format(img_path))
        y = model(pixs)
        feat_array.append(y[0])
    
    print(feat_array)
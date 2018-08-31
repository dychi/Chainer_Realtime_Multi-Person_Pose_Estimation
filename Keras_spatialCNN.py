# coding: utf-8
import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse

from keras.layers import Concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keras spatial CNN')
    parser.add_argument('--imgs_dir')
    parser.add_argument('--list_file')
    args = parser.parse_args()

    # VGG16
    vgg16 = VGG16(include_top=True, weights='imagenet')
    # without last layer
    vgg16.layers.pop()
    inp = vgg16.input
    out = [vgg16.layers[-1].output]
    model = Model(inp, out)

    # read image list 
    img_list_file = pd.read_csv(args.list_file, sep=',', usecols=[1,2,])
    imgs_name = img_list_file["Img_name"]

    num = 2
    feat_array_0 = []
    feat_array_1 = []
    for i, img_path in enumerate(imgs_name):
        if i == 2:
            break
        # player 0
        img_0 = image.load_img(args.imgs_dir + '/0/{}'.format(img_path), target_size=(224,224))
        x_0 = image.img_to_array(img_0)
        x_0 = np.expand_dims(x_0, axis=0)
        x_0 = preprocess_input(x_0)
        feature_0 = model.predict(x_0)
        print(type(feat_array_0))
        print(feature_0[0].shape)
        feat_array_0.append(feature_0[0])
        
        # player 1
        img_1 = image.load_img(args.imgs_dir + '/1/{}'.format(img_path), target_size=(224,224))
        x_1 = image.img_to_array(img_1)
        x_1 = np.expand_dims(x_1, axis=0)
        x_1 = preprocess_input(x_1)
        feature_1 = model.predict(x_1)
        feat_array_1.append(feature_1[0])
    
    np_feat_0 = np.array(feat_array_0)
    np_feat_1 = np.array(feat_array_1)
    print(np_feat_1.shape)
    both = np.concatenate([np_feat_0, np_feat_1], axis=1)
    print(both.shape)
    print(both)

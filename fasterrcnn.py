# coding: utf-8
import sys
import os
import argparse
import matplotlib.pyplot as plt

import chainer

import chainercv
from chainercv import utils
from chainercv.links import FasterRCNNVGG16
from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names


chainer.cuda.get_device_from_id(0).use()

def refine_bbox(bbox, ratio):
    top_h = bbox[0]
    top_w = bbox[1]
    bottom_h = bbox[2]
    bottom_w = bbox[3]
    gravity_h = (top_h + bottom_h) / 2
    gravity_w = (top_w + bottom_w) / 2
    height = abs(gravity_h - top_h)
    weight = abs(gravity_w - top_w)
    bbox[0] = gravity_h - height * ratio
    bbox[1] = gravity_w - weight * ratio
    bbox[2] = gravity_h + height * ratio
    bbox[3] = gravity_w + weight * ratio
    return bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img')
    parser.add_argument('--ratio', type=float, default=1.2)
    args = parser.parse_args()
    
    model = FasterRCNNVGG16(n_fg_class=len(voc_bbox_label_names), pretrained_model='voc07')
    model.to_gpu()
    img = utils.read_image(args.img, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]
    if len(bboxes) >= 0:
        for i in range(len(bboxes)):
            bbox[i] = refine_bbox(bbox[i], args.ratio)
    vis_bbox(img, bbox, label_names=voc_bbox_label_names)
    plt.show()


# coding: utf-8

# In[1]:

import cv2
import sys
import numpy as np
import argparse
import chainer
from entity import params
from pose_detector import PoseDetector, draw_person_pose
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

def select_region(image):
    if len(image.shape) == 3:
        high, wid, ch = image.shape
    else:
        high, wid = image.shape

    # make area
    bottom_left  = [wid*0.1, high*1]
    top_left     = [wid*0.30, high*0.3]
    bottom_right = [wid*0.9, high*1]
    top_right    = [wid*0.7, high*0.3]
    # polygons 
    poly = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, poly, 255)
    else:
        # in case of channel=3
        cv2.fillPoly(mask, poly, (255,)*mask.shape[2])

    return cv2.bitwise_and(image, mask), mask


# In[3]:

#load model
pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=-1, precise=True)


# In[4]:

# read image
img = cv2.imread('../pyfiles/dataset/youtube_baun/img_08310.png')
# select detection area
img, mask = select_region(img)
# inference
poses, scores = pose_detector(img)
#res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, poses), 0.4, 0)
img = draw_person_pose(img, poses)


# get unit_length 
unit_length = pose_detector.get_unit_length(poses)

# detect person
cropped_person_img, bbox = pose_detector.crop_person(img, poses, unit_length) 
if cropped_person_img is not None:
    # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
    crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] #bbox=(x_lefttop,y)

# In[82]:
# recognize players (0:bottom, 1:top)
num_person = poses.shape[0]

# each person detected
for i, pose in enumerate(poses[0:2]): # choose only 2 persons
    unit_length = pose_detector.get_unit_length(pose)
    # detect person
    cropped_person_img, bbox = pose_detector.crop_person(img, pose, unit_length) 
    if cropped_person_img is not None:
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] #bbox=(x_lefttop,y)
        cv2.imwrite('./data/crop_{0:02d}.png'.format(i), crop_img)



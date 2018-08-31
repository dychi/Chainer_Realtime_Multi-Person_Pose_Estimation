# coding: utf-8
import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import pickle

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Chainer_Realtime_Multi-Person_Pose_Estimation')
from badminton_pose_detector import PoseDetector

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int)

args = parser.parse_args()

#---------------- OpenPose --------------------
MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/models/coco_posenet.npz'
pose_detector = PoseDetector(arch='posenet', weights_file=MODEL_PATH, device=0)


#---------------- Read Images --------------------
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../badminton_action_recognition_using_pose_estimation/datasets'
num = args.num
labels = pd.read_csv(DATASETS_DIR + '/match_number.txt')
name = labels.dir_name[num-1]
labels_list = pd.read_csv(DATASETS_DIR + '/match_{0}/{1}_feature_images.txt'.format(num, name), usecols=[1,2])
img_names = labels_list.Img_name
print('processing {} and images length: {} for each player'.format(name, len(img_names)))
# Initialize Pose Vector
pose_0 = []
pose_1 = []
zeros = np.zeros((1,36))
for i, path in enumerate(img_names):
        #---- Player 0 ----
        try:
            img_path = DATASETS_DIR + '/match_{0}/0/{1}'.format(num, path)
            img = cv2.imread(img_path)
            poses, scores = pose_detector(img)
            pose = poses[0][:,:2].reshape((1,-1), order='F')
            pose_0.append(pose[0])
        except FileNotFoundError:
            print('Player 0 error at {}'.format(path))
            pose_0.append(zeros[0])
        except IndexError:
            print('Player 0 at {} not found poses'.format(path))
            pose_0.append(zeros[0])
        except AttributeError:
            print('Player 0 at {} may not be image'.format(path))
            pose_0.append(zeros[0])

        #---- Player 1 ----
        try:
            img_path = DATASETS_DIR + '/match_{0}/1/{1}'.format(num, path)
            img = cv2.imread(img_path)
            poses, scores = pose_detector(img)
            pose = poses[0][:,:2].reshape((1,-1), order='F')
            pose_1.append(pose[0])
        except FileNotFoundError:
            print('Player 1 error at {}'.format(path))
            pose_1.append(zeros[0])
        except IndexError:
            print('Player 1 at {} not found poses'.format(path))
            pose_1.append(zeros[0])
        except AttributeError:
            print('Player 1 at {} may not be image'.format(path))
            pose_1.append(zeros[0])



Poses = np.hstack((pose_0, pose_1))
Poses_shape = Poses.shape
print('End at {0} and {1}'.format(path, Poses_shape))

#---------------- Read Images --------------------
with open(DATASETS_DIR + '/match_{}/PoseFeature_{}.pkl'.format(num, name), 'wb') as f:
    pickle.dump(Poses, f)
print('Save Done')

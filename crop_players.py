import cv2
import os
import sys
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.append('../')
from original.entity import params, JointType
from badminton_pose_detector import PoseDetector, draw_person_pose

# define infer area
def select_region(image):
    if len(image.shape) == 3:
        high, wid, ch = image.shape
    else:
        high, wid = image.shape
        
    # define select areas
    bottom_left = [wid*0.1, high*1]
    top_left    = [wid*0.30, high*0.3]
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

# code from openpose
def compute_limbs_length(joints): # limbs_len is something wrong
    limbs = []
    limbs_len = np.zeros(len(params["limbs_point"])) # 19 points
    for i, joint_indices in enumerate(params["limbs_point"]):
        if joints[joint_indices[0]] is not None and joints[joint_indices[1]] is not None: # 鼻or首があるか確認する
            limbs.append([joints[joint_indices[0]], joints[joint_indices[1]]])
            limbs_len[i] = np.linalg.norm(joints[joint_indices[1]][:-1] - joints[joint_indices[0]][:-1])
        else:
            limbs.append(None)

    return limbs_len, limbs

def compute_unit_length(limbs_len): # 鼻首の長さを優先しない
    unit_length = 0
    base_limbs_len = limbs_len[[3, 0, 13, 9]] # (首左腰、首右腰、肩左耳、肩右耳)の長さの比率(このどれかが存在すればこれを優先的に単位長さの計算する)
    non_zero_limbs_len = base_limbs_len > 0
    if len(np.nonzero(non_zero_limbs_len)[0]) > 0:
        limbs_len_ratio = np.array([2.2, 2.2, 0.85, 0.85])
        unit_length = np.sum(base_limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])
    else:
        limbs_len_ratio = np.array([2.2, 1.7, 1.7, 2.2, 1.7, 1.7, 0.6, 0.93, 0.65, 0.85, 0.6, 0.93, 0.65, 0.85, 1, 0.2, 0.2, 0.25, 0.25]) # 鼻首を1としている
        non_zero_limbs_len = limbs_len > 0
        unit_length = np.sum(limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])

    return unit_length

def get_unit_length(person_pose):
    limbs_length, limbs = compute_limbs_length(person_pose)
    unit_length = compute_unit_length(limbs_length)

    return unit_length, limbs_length

# define crop person
def crop_person(img, person_pose, unit_length):
    top_joint_priority = [4, 5, 6, 12, 16, 7, 13, 17, 8, 10, 14, 9, 11, 15, 2, 3, 0, 1, sys.maxsize]
    bottom_joint_priority = [9, 6, 7, 14, 16, 8, 15, 17, 4, 2, 0, 5, 3, 1, 10, 11, 12, 13, sys.maxsize]

    top_joint_index = len(top_joint_priority) - 1
    bottom_joint_index = len(bottom_joint_priority) - 1
    left_joint_index = 0
    right_joint_index = 0
    top_pos = sys.maxsize
    bottom_pos = 0
    left_pos = sys.maxsize
    right_pos = 0

    for i, joint in enumerate(person_pose):
        if joint[2] > 0:
            if top_joint_priority[i] < top_joint_priority[top_joint_index]:
                top_joint_index = i
            elif bottom_joint_priority[i] < bottom_joint_priority[bottom_joint_index]:
                bottom_joint_index = i
            if joint[1] < top_pos:
                top_pos = joint[1]
            elif joint[1] > bottom_pos:
                bottom_pos = joint[1]

            if joint[0] < left_pos:
                left_pos = joint[0]
                left_joint_index = i
            elif joint[0] > right_pos:
                right_pos = joint[0]
                right_joint_index = i

    top_padding_ratio = [0.9, 1.9, 1.9, 2.9, 3.7, 1.9, 2.9, 3.7, 4.0, 5.5, 7.0, 4.0, 5.5, 7.0, 0.7, 0.8, 0.7, 0.8]
    bottom_padding_ratio = [6.9, 5.9, 5.9, 4.9, 4.1, 5.9, 4.9, 4.1, 3.8, 2.3, 0.8, 3.8, 2.3, 0.8, 7.1, 7.0, 7.1, 7.0]

    left = (left_pos - 0.3 * unit_length).astype(int)
    right = (right_pos + 0.3 * unit_length).astype(int)
    top = (top_pos - top_padding_ratio[top_joint_index] * unit_length).astype(int)
    bottom = (bottom_pos + bottom_padding_ratio[bottom_joint_index] * unit_length).astype(int)
    bbox = (left, top, right, bottom)

    cropped_img = pose_detector.crop_image(img, bbox)
    return cropped_img, bbox

# 矩形内の面積を求める
def get_bbox_area(bbox):
    width = abs(bbox[0] - bbox[2])
    hight = abs(bbox[1] - bbox[3])
    area = width * hight
    return area

# 矩形内の重心座標を取得する
def get_centerof_bbox(bbox): # should be (x,y)
    left_top = np.array([bbox[0], bbox[1]])
    right_bottom = np.array([bbox[2], bbox[3]])
    center_point = (left_top + right_bottom)//2
    return center_point

# 選手の位置は合っているが矩形の大きさがおかしいので、今のフレームの重心を中心にして前フレームのbboxの大きさを切り取る
def get_new_bbox(current_bbox_center, fixed_bbox_height, current_bbox_pos, current_pose):
    # copy array
    new_bbox = np.copy(current_bbox_pos)
    # define new_bbox with pose
    # pose_width = (
    # pose_height = 

    half_hight = fixed_bbox_height/2*1.2
    half_width = abs(current_bbox_pos[0] - current_bbox_pos[2])/2
    # bbox height
    new_bbox[1] = current_bbox_center[1] - half_hight
    new_bbox[3] = current_bbox_center[1] + half_hight
    # bbox width
    new_bbox[0] = current_bbox_center[0] - half_width
    new_bbox[2] = current_bbox_center[0] + half_width 
    return new_bbox

def modify_bbox(multi_person_poses, pose_num, image, img_name, fixed_bbox_height, previous_bbox):
    unit, limb_length = get_unit_length(multi_person_poses[pose_num])
    cropped_img, bbox = crop_person(image, multi_person_poses[pose_num], unit)
    current_bbox_center = get_centerof_bbox(bbox)
    current_bbox_height = abs(bbox[1] - bbox[3]) 
    # 大きさには対応しているかどうか
    if (current_bbox_height > fixed_bbox_height*1.4): # bboxが大きすぎる
        # 前フレームのbboxと比較して大きすぎたら、今のフレームの重心を中心に前フレームと同じ大きさのbboxを使う
        new_bbox = get_new_bbox(current_bbox_center, fixed_bbox_height, previous_bbox, multi_person_poses[pose_num])
        try:
            cropped_img = pose_detector.crop_image(image, new_bbox)
        except:
            print('dimention error at {0}, bbox: {1}'.format(img_name, new_bbox))
            # continue
        print('Bbox was too big in the image: {}'.format(img_name))
        previous_bbox = new_bbox
    else: # bboxが正しい大きさ
        # bboxの位置を更新する
        previous_bbox = bbox

    return cropped_img, previous_bbox

parser = argparse.ArgumentParser(description='Crop person')
parser.add_argument('--img_dir', '-d', help='original image dir')
# parser.add_argument('--out_dir', '-o')
parser.add_argument('--pose_num', '-pn', type=int)
parser.add_argument('--feat_file', '-ff', help='featured image list txt file')
# parser.add_argument('weights', help='weidths file path')
# parser.add_argument('--gpu', '-g', type=int, default=-1)
args = parser.parse_args()


#load model
pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0, precise=True)
print("load model done!")

# 有効なフレームのリストのテキストファイルを読み込む
imgs_dir = pd.read_csv(args.feat_file, sep=',', usecols=[1,2])

# bboxの初期化
previous_bbox = np.zeros(4)
# 最初のフレーム。initializer
first_frame = imgs_dir["Img_name"][0]
# read firstframe
img = cv2.imread(args.img_dir + '/{}'.format(first_frame))
img_copy = img.copy()
play_region_img, mask = select_region(img_copy)
multi_poses, scores = pose_detector(play_region_img)
# 全パーツの平均の座標でポーズをソートする ⇨ pose_num = 0:bottom, 1:top player
ave_pose = np.average(multi_poses[:], axis=1)
multi_person_poses = multi_poses[np.argsort(ave_pose[:,1])[::-1]]

# write both players of first frame
## player0
unit_0, limb_length_0 = get_unit_length(multi_person_poses[0])
cropped_img_0, bbox_0 = crop_person(img, multi_person_poses[0], unit_0)
print('Done first frame player_0')
previous_bbox_0 = list(bbox_0)
# write image
cv2.imwrite('./data/0/{}'.format(first_frame), cropped_img_0)
previous_bbox_center_0 = get_centerof_bbox(bbox_0)
fixed_bbox_height_0 = abs(bbox_0[1] - bbox_0[3])

## player1
unit_1, limb_length_1 = get_unit_length(multi_person_poses[1])
cropped_img_1, bbox_1 = crop_person(img, multi_person_poses[1], unit_1)
print('Done first frame player_1')
previous_bbox_1 = list(bbox_1)
# write image
cv2.imwrite('./data/1/{}'.format(first_frame), cropped_img_1)
previous_bbox_center_1 = get_centerof_bbox(bbox_1)
fixed_bbox_height_1 = abs(bbox_1[1] - bbox_1[3])

for i, img_name in enumerate(imgs_dir["Img_name"]):
    img = cv2.imread(args.img_dir + '/{}'.format(img_name))
    img_copy = img.copy()
    play_region_img, mask = select_region(img_copy)
    multi_poses, scores = pose_detector(play_region_img)
    # もしposes配列に一つもなければスキップする
    if (len(multi_poses)==0):
        print('Skipped at {}'.format(img_name))
        continue
    # 全パーツの平均の座標でポーズをソートする ⇨ pose_num = 0:bottom, 1:top player
    ave_pose = np.average(multi_poses[:], axis=1)
    multi_person_poses = multi_poses[np.argsort(ave_pose[:,1])[::-1]]
    # try-except
    try:
        cropped_img_0, pre_bbox_0 = modify_bbox(multi_person_poses, 0, img, img_name, fixed_bbox_height_0, previous_bbox_0)
        cropped_img_1, pre_bbox_1 = modify_bbox(multi_person_poses, 1, img, img_name, fixed_bbox_height_1, previous_bbox_1)
    except:
        continue
    cv2.imwrite('./data/0/{}'.format(img_name), cropped_img_0)
    cv2.imwrite('./data/1/{}'.format(img_name), cropped_img_1)

print('Done!')

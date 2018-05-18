import cv2
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

sys.path.append('../')
from original.entity import params, JointType
from badminton_pose_detector import PoseDetector, draw_person_pose


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
def get_new_bbox(current_bbox_center, previous_bbox):
    new_bbox = np.copy(previous_bbox)
    half_width = abs(previous_bbox[0] - previous_bbox[2])/2*1.2
    half_hight = abs(previous_bbox[1] - previous_bbox[3])/2*1.2
    new_bbox[0] = current_bbox_center[0] - half_width
    new_bbox[1] = current_bbox_center[1] - half_hight
    new_bbox[2] = current_bbox_center[0] + half_width
    new_bbox[3] = current_bbox_center[1] + half_hight
    return new_bbox


parser = argparse.ArgumentParser(description='Crop person')
parser.add_argument('--img_dir', '-d', help='original image dir')
# parser.add_argument('--out_dir', '-o')
parser.add_argument('--pose_num', '-pn', type=int)
# parser.add_argument('weights', help='weidths file path')
# parser.add_argument('--gpu', '-g', type=int, default=-1)
args = parser.parse_args()


#load model
pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0, precise=True)
print("load model done!")


# for 文でループ回す、一つ前のフレームを保持する
imgs_dir = pd.read_csv('./feature_images.txt', sep=',', usecols=[1,2])

previous_bbox = 0
for i, img_name in enumerate(imgs_dir["Img_name"]):
    img = cv2.imread(args.img_dir + '/{}'.format(img_name))
    img_copy = img.copy()
    play_region_img, mask = select_region(img_copy)
    multi_poses, scores = pose_detector(play_region_img)
    # 全パーツの平均の座標でポーズをソートする ⇨ 0:bottom, 1:top player
    ave_pose = np.average(multi_poses[:], axis=1)
    multi_person_poses = multi_poses[np.argsort(ave_pose[:,1])[::-1]]
    
    # 0:bottom, 1:top plaiyer
    pose_num = args.pose_num
    # pose_numが変わってもディレクトリは変わらないようにするため
    player_dir = pose_num
    
    high, wid, ch = img.shape
    # もしposes配列に一つもなければスキップする
    if (len(multi_poses)==0):
        continue
    
    unit, limb_length = get_unit_length(multi_person_poses[pose_num])
    cropped_img, bbox = crop_person(img, multi_person_poses[pose_num], unit)
    
    # 最初のループは飛ばす, initializer
    if i == 0:
        print('Skipped first roop')
        previous_bbox = list(bbox)
        cv2.imwrite('./data/{0}/{1}'.format(player_dir, img_name), cropped_img)
        previous_bbox_center = get_centerof_bbox(bbox)
        continue
        
    # 前フレームのbboxの座標と比較して離れすぎていたら前のフレームのbboxを利用する
    current_bbox_center = get_centerof_bbox(bbox)
    diffence = np.abs(previous_bbox_center - current_bbox_center)
    diffence_length = np.linalg.norm(diffence)
    bbox_width = abs(bbox[0] - bbox[2])
    
    # 選手位置がずれていないか
    if (diffence_length > bbox_width): # 選手位置が大きくずれている
        print('Detected different player: {}'.format(img_name))
        # ---------ここで違う姿勢を検出してしまったときの処理必要---------
        if pose_num == 0:
            pose_num += 1
        else:
            pose_num -= 1
        unit, limb_length = get_unit_length(multi_person_poses[pose_num])
        cropped_img, bbox = crop_person(img, multi_person_poses[pose_num], unit)
        current_bbox_center = get_centerof_bbox(bbox)
        new_bbox = get_new_bbox(current_bbox_center, previous_bbox)
        # selected area 以外で検出されて人物を切り取ろうとするとエラーが発生する
        img = cv2.imread(args.img_dir + '/{}'.format(img_name))
        cropped_img = pose_detector.crop_image(img, new_bbox)
        
    else: # 選手位置がずれていない
        # 矩形の面積を求める
        previous_bbox_area = get_bbox_area(previous_bbox)
        current_bbox_area = get_bbox_area(bbox)
        # 選手の位置がずれていなかったら重心の更新
        previous_bbox_center = np.copy(current_bbox_center)
        # 大きさには対応しているかどうか
        if (current_bbox_area > previous_bbox_area*2): # bboxが大きすぎる
            # 前フレームのbboxと比較して大きすぎたら、今のフレームの重心を中心に前フレームと同じ大きさのbboxを使う
            new_bbox = get_new_bbox(current_bbox_center, previous_bbox)
            img = cv2.imread(args.img_dir + '/{}'.format(img_name))
            cropped_img = pose_detector.crop_image(img, new_bbox)
            print('Bbox was too big in the image: {}'.format(img_name))
        else: # bboxが正しい大きさ
            # bboxの位置を更新する
            previous_bbox = bbox
            
    cv2.imwrite('./data/{0}/{1}'.format(player_dir, img_name), cropped_img)

print('Done!')

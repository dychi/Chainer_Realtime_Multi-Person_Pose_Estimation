import cv2
import os
import sys
import pandas as pd
import numpy as np
import argparse
import pickle

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

def get_fix_bbox(pose):
    # 重心を計算
    pose_xy = pose[:, :2]
    nonzero = pose_xy.nonzero()
    current_bbox_center = pose_xy[nonzero].reshape(-1, 2).mean(axis=0)
    # 固定サイズで切り取り
    height = 454
    width = 340
    new_bbox = np.zeros(4).astype(np.int64) # typeを指定しないとcrop_imageでslice errorが起きる
    new_bbox[0] = current_bbox_center[0] - (width // 2)
    new_bbox[1] = current_bbox_center[1] - (height // 2)
    new_bbox[2] = current_bbox_center[0] + (width // 2)
    new_bbox[3] = current_bbox_center[1] + (height // 2)
    return new_bbox, current_bbox_center

def get_bbox_center(pose):
    # 重心を計算
    pose_xy = pose[:, :2]
    nonzero = pose_xy.nonzero()
    bbox_center = pose_xy[nonzero].reshape(-1, 2).mean(axis=0)
    return bbox_center

def CalcBbox(ordered_poses:list, previous_bbox:list, img_copy):
    # Crop Previous Image
    prev_cropped_img = pose_detector.crop_image(img_copy, previous_bbox)
    prev_hist = CalcHist(prev_cropped_img)
    
    # Compare Current Image's histgram
    pose_len = len(ordered_poses)
    res = []
    bbox = []
    for i in range(pose_len):
        new_bbox, bbox_center = get_fix_bbox(ordered_poses[i])
        cropped_img = pose_detector.crop_image(img_copy, new_bbox)
        hist = CalcHist(cropped_img)
        score = cv2.compareHist(prev_hist, hist, 0)
        res.append(score)
        bbox.append(new_bbox)
    ### -------- histgramを比較 --------------
    max_index = np.argmax(res)
    correct_bbox = bbox[max_index]
    if res[max_index] < 0.7:
        correct_bbox = previous_bbox
    return correct_bbox, max_index

def CalcHist(cropped_img):
# Histgram
    color = ('b','g','r')
    Hist = []
    for i,col in enumerate(color):
        histr = cv2.calcHist([cropped_img],[i],None,[256],[0,256])
        Hist.append(histr[:])
    return np.squeeze(Hist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop person')
    parser.add_argument('--num', type=int)
    args = parser.parse_args()

    #load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0, precise=True)
    print("load model done!")

    # DATASETS
    DATASETS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../badminton_action_recognition_using_pose_estimation/datasets'
    # Match Number and Name
    num = args.num
    labels = pd.read_csv(DATASETS_DIR + '/match_number.txt')
    name = labels.dir_name[num-1]

    # 有効なフレームのリストのテキストファイルを読み込む
    imgs_dir = pd.read_csv(DATASETS_DIR + '/match_{0}/{1}_feature_images.txt'.format(num, name), sep=',', usecols=[1,2])
    
    # ====== 初期化 =======
    # 最初のフレーム。initializer
    first_frame = imgs_dir["Img_name"][0]
    # read firstframe
    img = cv2.imread(DATASETS_DIR + '/match_{0}/{1}/{2}'.format(num, name, first_frame))
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    play_region_img, mask = select_region(img_copy)
    multi_poses, scores = pose_detector(play_region_img)
    # ポーズの配列で検出された関節点の数が8より少なければドロップする
    nonzero_index = [i for i, pose in enumerate(multi_poses[:,:,0]) if np.count_nonzero(pose) > 8]
    # 全パーツの高さの座標でポーズをソートする ⇨ pose_num = 0:bottom, 1:top player
    max_pose = np.max(multi_poses[nonzero_index], axis=1)
    ordered_poses = multi_poses[np.argsort(max_pose[:,1])[::-1]]

    ## --------- player0 ---------
    first_bbox_0, first_bbox_center_0 = get_fix_bbox(ordered_poses[0])
    cropped_img_0 = pose_detector.crop_image(img, first_bbox_0)
    print('Done first frame player_0', first_bbox_0)
    previous_bbox_0 = first_bbox_0
    # write image
    cv2.imwrite(DATASETS_DIR + '/match_{0}/0/{1}'.format(num, first_frame), cropped_img_0)

    ## --------- player1 ---------
    first_bbox_1, first_bbox_center_1 = get_fix_bbox(ordered_poses[1])
    cropped_img_1 = pose_detector.crop_image(img, first_bbox_1)
    print('Done first frame player_1')
    previous_bbox_1 = first_bbox_1
    # write image
    cv2.imwrite(DATASETS_DIR + '/match_{0}/1/{1}'.format(num, first_frame), cropped_img_1)
    
    # ========= iterator ==========
    # Prepare Extract Index
    extract_index =[]
    imgs_num = imgs_dir['Img_name'].str.extract('(.+)_(.+)\.(.+)')[1]
    for i, num_str in enumerate(imgs_num):
        if num_str == list(imgs_num)[-1]:
            break
        elif (int(imgs_num[i+1]) - int(num_str)) > 2:
            # カメラアングルが変わる前のインデックスを保存して、下のprevious_bboxの更新時に利用する
            extract_index.append(i)
    
    ## Pose用の配列を準備する
    pose_0 = []
    pose_1 = []
    pose_dic0 = {}
    pose_dic1 = {}

    for i, img_name in enumerate(imgs_dir["Img_name"]):
        # Skip 
        #if i > 3 and i < 130:
            #continue

        img = cv2.imread(DATASETS_DIR + '/match_{0}/{1}/{2}'.format(num, name, img_name))
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        play_region_img, mask = select_region(img_copy)
        multi_poses, scores = pose_detector(play_region_img)
        # ポーズの配列で検出された関節点の数が8より少なければドロップする
        nonzero_index = [i for i, pose in enumerate(multi_poses[:,:,0]) if np.count_nonzero(pose) > 8]
        
        # もしposes配列に一つもなければスキップする
        if (len(nonzero_index) <= 1):
            print('Skipped at {}'.format(img_name))
            continue
        
        # 全パーツの高さの最大座標でポーズをソートする ⇨ pose_num = 0:bottom, 1:top player
        max_pose = np.max(multi_poses[nonzero_index], axis=1)
        ordered_poses = multi_poses[np.argsort(max_pose[:,1])[::-1]]
        
        # Calculate Correct Bbox
        correct_bbox_0, index_0 = CalcBbox(ordered_poses, previous_bbox_0, img_copy)
        correct_bbox_1, index_1 = CalcBbox(ordered_poses, previous_bbox_1, img_copy)
        
        # Get Bbox Center
        bbox_center_0 = get_bbox_center(ordered_poses[index_0])
        bbox_center_1 = get_bbox_center(ordered_poses[index_1])

        # Condition for Bounding Box
        if bbox_center_0[1] <= bbox_center_1[1]:
            print('Index0: {}, Index1: {}'.format(index_0, index_1))
            # ポーズの重心から固定サイズのbboxを切り取る
            correct_bbox_0, bbox_center_0 = get_fix_bbox(ordered_poses[0])
            correct_bbox_1, bbox_center_1 = get_fix_bbox(ordered_poses[1])
        


        # Crop Image
        cropped_img_0 = pose_detector.crop_image(img, correct_bbox_0)
        cropped_img_1 = pose_detector.crop_image(img, correct_bbox_1)

        cv2.imwrite(DATASETS_DIR + '/match_{0}/0/{1}'.format(num, img_name), cropped_img_0)
        cv2.imwrite(DATASETS_DIR + '/match_{0}/1/{1}'.format(num, img_name), cropped_img_1)

        print(img_name)
        ## POSE FEATURE
        pose0 = ordered_poses[index_0][:,:2].reshape((1,-1), order='F')
        pose1 = ordered_poses[index_1][:,:2].reshape((1,-1), order='F')
        pose_0.append(pose0[0])
        pose_1.append(pose1[0])
        pose_dic0[img_name] = pose0[0]
        pose_dic1[img_name] = pose1[0]

        # Update bbox
        if i in extract_index:
            previous_bbox_0 = first_bbox_0
            previous_bbox_1 = first_bbox_1
        else:
            previous_bbox_0 = correct_bbox_0
            previous_bbox_1 = correct_bbox_1
        
    # Save PoseFeature
    Poses = np.hstack((pose_0, pose_1))
    Poses_shape = Poses.shape
    print('End at {0} and {1}'.format(img_name, Poses_shape))
    with open(DATASETS_DIR + '/match_{0}/PoseFeature_scrach_{1}.pkl'.format(num, name), 'wb') as f:
        pickle.dump(Poses, f)
    
    print('Done!')

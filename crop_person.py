import cv2
import numpy as np
import argparse
import chainer
from entity import params
from pose_detector import PoseDetector, draw_person_pose


chainer.using_config('enable_backprop', False)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (nagative value indicates CPU)')
    args = parser.parse_args()


    #load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)

    # read image
    img = cv2.imread(args.img)
    # select detection area
    img, mask = select_region(img)
    cv2.imwrite('crop.png', img)

    # inference
    print("Estimating pose...")
    person_pose_array = pose_detector(img)
    res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

    # each person detected
    for i, person_pose in enumerate(person_pose_array):
         unit_length = pose_detector.get_unit_length(person_pose)
         print(unit_length)
       
        # detect person
         cropped_person_img, bbox = pose_detector.crop_person(img, person_pose, unit_length) 
         if cropped_person_img is not None:
             # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
             crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] #bbox=(x_lefttop,y)
             cv2.imwrite('./data/crop_{0:02d}.png'.format(i), crop_img)
             print(bbox)

    print('Saving result into crop_result.png...')
    #cv2.imwrite('crop_result.png', res_img)


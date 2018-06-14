import os
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='output feature annotation')
parser.add_argument('--txt', help='annotation txt file')
parser.add_argument('--label', help='label text')
parser.add_argument('--imlist', help='directory of images')
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--name', help='name of file')
args = parser.parse_args()

# read annotation file
anno_text = pd.read_csv(args.txt, header=None, sep='', delimiter='\t', usecols=[2,3,5], names=["start", "end", "label"])
# read action label directory
action_label = pd.read_csv(args.label, header=None, sep='\t')
action_index = {l:i for i, l in enumerate(action_label[0])}

# convert labels "string" to "int"
anno_tmp = np.ones(len(anno_text["label"]))
for i in range(len(anno_tmp)):
    anno_tmp[i] = action_index[anno_text["label"][i]]
label_num = pd.Series(anno_tmp, name="label_num")
anno_num = pd.concat([anno_text.loc[:, ["start", "end"]], label_num], axis=1)

# prepare for image name list
start_time = anno_num["start"]//40
end_time = anno_num["end"]//40
# read image directory
imglist = os.listdir(args.imlist)
img_list = pd.Series(imglist).str.extract('(.+)_(.+)\.(.+)', expand=True)
# sort
frames = np.array(img_list.sort_values(1)[1], np.int)

# label frames
frame_label = np.zeros(len(frames))
for i, frame in enumerate(frames):
    idx = np.nonzero((start_time <= frame) & (frame < end_time))[0]
    if len(idx)>0:
        frame_label[i] = anno_num["label_num"][idx]
    else:
        frame_label[i] = "NaN"

# concat 
pd_img = pd.Series(frames, name="Frame")
pd_label = pd.Series(frame_label, name="Label")
pd_img_label = pd.concat([pd_img, pd_label], axis=1)
# drop NaN
feature_labels = pd_img_label.dropna()
# create labeled image list
labeled_imglist = []
for i, num in enumerate(feature_labels["Frame"]):
    labeled_imglist.append("img_{0:06d}.jpg".format(num))
feature_label = list(feature_labels["Label"])
# concat 
df_list = pd.Series(labeled_imglist, name="Img_name")
df_feat = pd.Series(feature_label, name="Labels")
df_labels = pd.concat([df_list, df_feat], axis=1)
# output text file 
df_labels.to_csv(args.outdir + '/{}_feature_images.txt'.format(args.name))

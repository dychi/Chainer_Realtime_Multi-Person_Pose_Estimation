import numpy as np
import pandas as pd
import os

ori_anno = pd.read_csv('../../Ano..', sep=" ", delimiter="\t", usecols=[2,3,5], header=None, names=["start", "end", "label"])
action_labels = pd.read_csv('./labels.txt')
index = {l:i for i,l in enumerate(labels[0])}

images_list = os.listdir('../Annotation/youtube')


labels_frame = 

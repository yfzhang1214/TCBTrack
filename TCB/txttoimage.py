import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import os
import numpy as np


data_path = "/home/yfzhang/TCBTrack/TCB/datasets/MOT20/train/"##
out_path = "/home/yfzhang/TCBTrack/TCB/vis/"
seqs = os.listdir(data_path)
for seq in sorted(seqs):
    if not ('FRCNN' in seq or 'MOT20' in seq):
        continue
    txt_path = os.path.join("/home/yfzhang/TCBTrack/TCB/YOLOX_outputs/mot20/track_results",seq)+'.txt'##
    seq_path = os.path.join(data_path, seq)
    img_path = os.path.join(seq_path, 'img1')
    per_out_path = os.path.join(out_path,seq)
    if not os.path.exists(per_out_path):
        os.makedirs(per_out_path)
    
    images = os.listdir(img_path)
    num_images = len([image for image in images if 'jpg' in image])
    image_range = [num_images//2,num_images-1]
    track_result = np.loadtxt(txt_path, dtype = np.float32, delimiter=',')
    count = int(0)
    for i in range(num_images):
        if i < image_range[0] or i > image_range[1]:
                    continue
        img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq,i+1)))
        while(count< len(track_result) and track_result[count][0]==i+1-num_images//2):
            tid = track_result[count][1]
            bbox = track_result[count][2:6]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])),
                            (int(tid)*101%255, int(tid)*89%255, int(tid)*103%255), 2)
            count+=1
            img = cv2.putText(img, '{}'.format(int(tid)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(per_out_path+"/{}.jpg".format(i+1-num_images//2),img)

    print(seq)
        
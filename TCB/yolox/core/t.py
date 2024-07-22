import os
import os
import numpy as np
import json
import cv2


path = "/data3/yfzhang/datasets/bdd100k/images/track/train"
video_id = 0
out = []
seqs = os.listdir(path)
for seq in sorted(seqs):
    img_path = os.path.join(path,seq)
    images = os.listdir(img_path)
    num_images = len([image for image in images if 'jpg' in image])
    video_id +=1
    out.append(num_images)
np.savetxt("/home/yfzhang/DanceTrack/ByteTrack_ReID/bdd100k.txt",np.array(out,dtype=int))


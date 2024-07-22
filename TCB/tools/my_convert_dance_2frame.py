import os
import numpy as np
import json
import cv2

DATA_PATH = 'datasets/dancetrack'
OUT_PATH = os.path.join(DATA_PATH,'val3')
#SPLITS = ['val2']
fr = 3
if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    data_path = os.path.join(DATA_PATH,'val')
    seqs = os.listdir(data_path)
    for seq in sorted(seqs):

        seq_path = os.path.join(data_path,seq)
        gt_path = os.path.join(seq_path,'gt/gt.txt')
        gt = np.loadtxt(gt_path, dtype = np.float32, delimiter=',')
        gt_out_path = os.path.join(OUT_PATH,seq)
        if not os.path.exists(gt_out_path):
            os.makedirs(gt_out_path)
            os.makedirs(os.path.join(gt_out_path,'gt'))
        gt_out = os.path.join(gt_out_path,'gt/gt.txt')
        fout = open(gt_out,'w')

        for o in gt:
            if(int(o[0])%fr==1):
                fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                            int(o[0])//fr+1, int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                            int(o[6]), int(o[7]), o[8]))  
        fout.close()
        print(seq)

        ini_out = os.path.join(gt_out_path,'')

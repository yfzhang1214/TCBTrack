import os
import numpy as np
import json
import cv2

DATA_PATH = 'datasets/mot'
#DATA_PATH = 'datasets/MOT20'
OUT_PATH = os.path.join(DATA_PATH,'val')

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    data_path = os.path.join(DATA_PATH,'train')
    out = {'images':[], 'annotations':[], 'videos':[], 
           'categories': [{'id':1, 'name':'pedestrain'}]}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue
        if not ('FRCNN' in seq or 'MOT20' in seq):
            continue
        video_cnt+=1 # video sequence number
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, 'img1')
        ann_path = os.path.join(seq_path, 'gt/gt.txt')
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])
        image_range = [num_images//2, num_images-1]
        output_gt = os.path.join(OUT_PATH, seq)
        if not os.path.exists(output_gt):
            os.makedirs(output_gt)
            os.makedirs(os.path.join(output_gt, 'gt'))
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq,i+1)))
            height, width = img.shape[:2]
            image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq,i+1),
                          'id': image_cnt + i + 1,#整个training set的image id，从1算起
                          'frame_id':i+1-image_range[0],
                          'prev_image_id': image_cnt + i if i>0 else -1,
                          'next_image_id': image_cnt + i + 2 if i<num_images - 1 else -1,
                          'video_id':video_cnt,
                          'height':height,
                          'width':width}
            image_info['id']=image_info['id']-num_images//2
            out['images'].append(image_info)
        print('{}: {} images'.format(seq,num_images))
        anns = np.loadtxt(ann_path, dtype = np.float32, delimiter=',')
        anns_out = np.array([anns[i] for i in range(anns.shape[0])
                             if int(anns[i][0])-1>=image_range[0] and
                             int(anns[i][0])-1<=image_range[1]], np.float32)
        anns_out[:,0]-=image_range[0]
        gt_out = os.path.join(output_gt, 'gt/gt.txt')
        fout = open(gt_out,'w')
        for o in anns_out:
            fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                        int(o[6]), int(o[7]), o[8]))  
        fout.close()


            


            

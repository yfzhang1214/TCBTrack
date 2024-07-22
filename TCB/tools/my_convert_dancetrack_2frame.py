import os
import numpy as np
import json
import cv2

DATA_PATH = 'datasets/dancetrack'
OUT_PATH = os.path.join(DATA_PATH,'annotations')
SPLITS = ['val3']
fr = 3###
if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH,'val')
        out_path = os.path.join(OUT_PATH,'val3.json')###
        out = {'images':[], 'annotations':[], 'videos':[], 
               'categories': [{'id':1, 'name':'dancer'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        id_curr = 1
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt+=1 #video sequence number
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])

            count = 0
            for i in range(0,num_images,fr):
                count+=1
                img = cv2.imread(os.path.join(data_path, '{}/img1/{:08d}.jpg'.format(seq, i + 1)))
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/img1/{:08d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + count,  # image number in the entire training set.
                              'frame_id': count,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + count-1 if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + count+1 if i < num_images - fr else -1,
                              'video_id': video_cnt,
                              'height': height,
                              'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq,num_images))

            anns = np.loadtxt(ann_path, dtype = np.float32, delimiter=',')
            anns_out = np.array([anns[i] for i in range(anns.shape[0]) if int(anns[i][0])%fr==1],np.float32)
            anns_out[:,0] = (anns_out[:,0]+fr-1)//fr
            gt_out = os.path.join(seq_path, 'gt/gt_3.txt')###
            fout = open(gt_out,'w')
            for o in anns_out:
                fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                            int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                            int(o[6]), int(o[7]), o[8]))  
            fout.close()#生成gt_train和gt_val
            max_id = -1
            for i in range(anns_out.shape[0]):
                frame_id = int(anns_out[i][0])
                track_id = int(anns_out[i][1])
                cat_id = int(anns_out[i][7])
                ann_cnt+=1
                category_id = 1
                #if not track_id==tid_last:
                #        tid_curr +=1
                #        tid_last = track_id
                max_id = max(max_id,track_id)
                ann = {'id':ann_cnt,
                       'category_id': category_id,
                       'image_id': image_cnt+frame_id,
                       'track_id':id_curr+track_id,
                       'bbox': anns_out[i][2:6].tolist(),#左上+宽高
                       'conf': float(anns_out[i][6]),
                       'iscrowd':0,
                       'area': float(anns_out[i][4] * anns_out[i][5])}
                out['annotations'].append(ann)
            print('{}: {} ann images'.format(seq, int(anns_out[:, 0].max())))
            image_cnt+=(num_images+fr-1)//fr
            id_curr+=max_id+1
            print(id_curr)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
import os
import os
import numpy as np
import json
import cv2

DATA_PATH = '/data3/yfzhang/datasets/bdd100k/images/track/'
ANN_PATH = '/data3/yfzhang/datasets/bdd100k/labels/box_track_20/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
SPLITS = ['train']
if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    for split in SPLITS:
        data_path = os.path.join(DATA_PATH,split)
        an_path = os.path.join(ANN_PATH, split)
        out_path = os.path.join(OUT_PATH,'{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': []}
        out['categories'] = [{'id':1, 'name':'pedestrian'},
                             {'id':2, 'name':'rider'},
                             {'id':3, 'name':'car'},
                             {'id':4, 'name':'truck'},
                             {'id':5, 'name':'bus'},
                             {'id':6, 'name':'train'},
                             {'id':7, 'name':'motorcycle'},
                             {'id':8, 'name':'bicycle'}]
        categories = {'pedestrian':1,
                      'rider':2,
                      'car':3,
                      'truck':4,
                      'bus':5,
                      'train':6,
                      'motorcycle':7,
                      'bicycle':8}
        seqs = os.listdir(data_path)
        image_cnt=0
        ann_cnt=0
        video_cnt=0
        id_curr=0
        trackid = np.zeros([1500000,],dtype=int)
        for seq in sorted(seqs):
            video_cnt += 1
            out['videos'].append({'id':video_cnt, 'filename':seq})
            img_path = os.path.join(data_path,seq)
            ann_path = os.path.join(an_path,'{}.json'.format(seq))
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])

            for i in range(num_images):
                img = cv2.imread(os.path.join(data_path, '{}/{}-{:07d}.jpg'.format(seq,seq,i+1)))
                height, width = img.shape[:2]
                image_info = {'file_name':'{}/{}-{:07d}.jpg'.format(seq,seq,i+1),
                              'id':image_cnt+i+1,
                              'frame_id':i+1,
                              'prev_image_id': image_cnt+i if i>0 else -1,
                              'next_image_id':image_cnt+i+2 if i<num_images-1 else -1,
                              'video_id':video_cnt,
                              'height':height,
                              'width':width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            if split != 'test':
                f = open(ann_path,'r')
                anns = json.loads(f.read())
                for i in range(len(anns)):
                    for j in range(len(anns[i]['labels'])):
                        if anns[i]['labels'][j]['category'] not in categories.keys():
                            continue
                        frame_id = anns[i]['frameIndex']+1
                        track_id = int(anns[i]['labels'][j]['id'])
                        if(trackid[track_id]==0):#not appeared
                            id_curr +=1
                            trackid[track_id]=id_curr
                            my_id = id_curr
                        else:
                            my_id = trackid[track_id]
                        cat_id = categories[anns[i]['labels'][j]['category']]
                        ann_cnt +=1
                        x1 = anns[i]['labels'][j]['box2d']['x1']
                        x2 = anns[i]['labels'][j]['box2d']['x2']
                        y1 = anns[i]['labels'][j]['box2d']['y1']
                        y2 = anns[i]['labels'][j]['box2d']['y2']
                        ann = {'id':ann_cnt,
                               'category_id':cat_id,
                               'image_id':image_cnt+frame_id,
                               'track_id':int(my_id),
                               'bbox':[x1,y1,x2-x1,y2-y1],
                               'conf':1,#没考虑occluded, truncated, crowd
                               'iscrowd':int(anns[i]['labels'][j]['attributes']['crowd']),
                               'area': float(x2-x1)*float(y2-y1)}
                        out['annotations'].append(ann)
                        
                print('{}: {} ann images'.format(seq, ann_cnt))
            image_cnt += num_images
        print(id_curr)
        print('loaded {} for {} images and {} samples'.format(split,len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))




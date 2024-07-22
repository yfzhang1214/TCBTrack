import os
import os
import numpy as np
import json
import cv2

ANN_PATH = '/data3/yfzhang/datasets/bdd100k/labels/box_track_20/'
OUT_PATH = '/data3/yfzhang/datasets/bdd100k/labels_with_ids/'
SPLITS = ['val']

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,{c},-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, class_ids in results:
            for tlwh, track_id, c_id in zip(tlwhs, track_ids, class_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, c=c_id)
                f.write(line)


if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        an_path = os.path.join(ANN_PATH, split)
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
        
        seqs = os.listdir(an_path)
        count = 0
        results = []
        gt_filename = os.path.join(OUT_PATH,'bdd_gt.txt')
        trackid = np.zeros([1500000,],dtype=int)
        id_curr=0
        for seq in sorted(seqs):
            video_name = seq.split('.')[0]
            ann_path = os.path.join(an_path,seq)
            f = open(ann_path,'r')
            anns = json.loads(f.read())
            for i in range(len(anns)):#frame_id
                gt_tlwhs = []
                gt_ids = []
                gt_cids = []
                for j in range(len(anns[i]['labels'])):#annotation
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
                    x1 = anns[i]['labels'][j]['box2d']['x1']
                    x2 = anns[i]['labels'][j]['box2d']['x2']
                    y1 = anns[i]['labels'][j]['box2d']['y1']
                    y2 = anns[i]['labels'][j]['box2d']['y2']
                    gt_tlwhs.append([x1,y1,x2-x1,y2-y1])
                    gt_ids.append(my_id)
                    gt_cids.append(cat_id)
                results.append((i+1+count,gt_tlwhs,gt_ids,gt_cids))

            count+=len(anns)
            print("{}/{}".format(count,len(seqs)))
        write_results(gt_filename, results)
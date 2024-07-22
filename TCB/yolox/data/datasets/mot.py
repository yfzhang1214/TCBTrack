import cv2
import numpy as np
from pycocotools.coco import COCO

import os
import random
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
import torch

class MOTDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()                # image ids, not track ids
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        try:
            self.nID = self.get_total_ids()         # TODO: total ids for reid classifier
        except:
            pass
    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]#to tlbr format
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]        # format: tlbr
            res[ix, 4] = cls                        # class id, 0 for person
            res[ix, 5] = obj["track_id"]            # track id

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        twoframe = False
        if(type(index)!= type(int(1)) and index.shape==torch.Size([2])):#twoframesampler
            #print(1)
            twoframe = True
            try:
                id1_ = self.ids[index[0]]
                id2_ = self.ids[index[1]]
            except:
                aaa=1
            res1, img_info1, file_name1 = self.annotations[index[0]]
            res2, img_info2, file_name2 = self.annotations[index[1]]
            img_file1 = os.path.join(self.data_dir, self.name, file_name1)
            img_file2 = os.path.join(self.data_dir, self.name, file_name2)
            img1 = cv2.imread(img_file1)
            img2 = cv2.imread(img_file2)
            assert img1 is not None
            assert img2 is not None

            return (img1,img2), (res1.copy(),res2.copy()), (img_info1,img_info2), (np.array([id1_]),np.array([id2_])), twoframe
            

        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        img = cv2.imread(img_file)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_]), twoframe

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id, twoframe = self.pull_item(index)
        if(twoframe==False):
            if self.preproc is not None:
                img, target = self.preproc(img, target, self.input_dim)
            return img, target, img_info, img_id
        else:
            if self.preproc is not None:
                pd = [random.randrange(2),random.randrange(2),random.randrange(2),random.randrange(2),random.randrange(2)]
                #pd = [0,0,0,0,0]
                img1, target1 = self.preproc(img[0],target[0],self.input_dim,pd)
                img2, target2 = self.preproc(img[1],target[1],self.input_dim,pd)
            return (img1,img2), (target1,target2), img_info, img_id

    # TODO: get total ids for each dataset, which is used in the classifier of reid branch
    def get_total_ids(self):
        max_id_each_img = []
        for annotation in self.annotations:     # tuple (3): (res, img_info, file_name),
            res = annotation[0]
            if len(res)==0:
                continue
            max_id_each_img.append(int(max(res[:, 5])))
        total_ids = max(max_id_each_img)         # TODO Need Check: ids start with 0，是0的话要+1;这个和cross entropy loss有关
        return total_ids+1

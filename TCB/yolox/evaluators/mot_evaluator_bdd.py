from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch
import cv2
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.my_byte_tracker_bdd import BYTETracker#################################################################修改！！！
#from yolox.tracker.byte_tracker_nokal import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import torch.nn.functional as F

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,{c},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, class_ids in results:
            for tlwh, track_id, c_id in zip(tlwhs, track_ids, class_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w,
                                          h=h, c=c_id)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class BDDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
            self,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor  # HalfTensor
        model = model.eval()
        if half:  # True
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        tracker = BYTETracker(self.args)  # yolox/tracker/byte_tracker.py
        ori_thresh = self.args.track_thresh  # 0.6
        count = 1
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            #im = cv2.imread(os.path.join("/data3/yfzhang/datasets/bdd100k/images/track/val/",info_imgs[4][0]))
            #vis_path = os.path.join("/data3/yfzhang/datasets/bdd100k/images/track/vis/",(info_imgs[4][0]).split("/")[0])
            #if not os.path.exists(vis_path):
            #    os.makedirs(vis_path)
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    
                    #params=[0.3,0.8,0.2,-100,0.1]
                    #print(tracker.num1, tracker.num3, tracker.num4)
                    tracker = BYTETracker(self.args,params=[0.7,0.9,0.1])
                    #BYTETracker(self.args)
                    

                imgs = imgs.type(tensor_type)
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs = model(imgs)  # [batchsize, all_anchors, 6], 6 for bbox + obj + cls
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)  # yolox/utils/boxes.py

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)       # TODO: no need fpr process for embeddings?
            data_list.extend(
                output_results)  # list, length of [batchsize], dict which keys is ['image_id', 'category_id', 'bbox', 'score', 'segmentation']

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)   # TODO: ReID. add 'id_feature'
                online_tlwhs = []
                online_ids = []
                online_cids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > self.args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        #online_cids.append(3)
                        online_cids.append(t.class_id)
                        bbox = t.tlbr
                #        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])),
                #            (int(bbox[2]), int(bbox[3])),
                #            (int(tid)*101%255, int(tid)*89%255, int(tid)*103%255), 2)
                #        im = cv2.putText(im, '{}'.format(int(tid)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                #cv2.imwrite(os.path.join("/data3/yfzhang/datasets/bdd100k/images/track/vis/",info_imgs[4][0]),im)
                # save results
                results.append((count, online_tlwhs, online_ids, online_cids))
            count = count+1

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        result_filename = "/home/yfzhang/DanceTrack/ByteTrack_ReID/YOLOX_outputs/my_bdd100k/track_results/bdd.txt"
        write_results(result_filename, results)

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
            )
            ]
        )
        print(time_info)
        return

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]  # get predicted bbox

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]  # get predicted class
            scores = output[:, 4] * output[:, 5]  # score = obj_conf * cls_conf
            for ind in range(bboxes.shape[0]):  # iteration over number of bboxes
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list
    def convert_scale(self,outputs,info_imgs):
        box_corner = torch.zeros((outputs.shape[0],4))
        scale = min(self.img_size[0] / float(info_imgs[0]), self.img_size[1] / float(info_imgs[1]))
        box_corner[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
        box_corner[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
        box_corner[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
        box_corner[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
        box_corner/=scale
        boxes = xyxy2xywh(box_corner)
        outputs[:, :4] = boxes[:, :4]
        return outputs

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            # from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

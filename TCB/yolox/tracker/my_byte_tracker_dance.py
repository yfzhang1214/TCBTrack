import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from cython_bbox import bbox_overlaps as bboxes_iou
#from yolox.utils import bboxes_iou

class STrack(BaseTrack):
    #shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=60):  # todo: ReID. add inputs of 'temp_feat', 'buffer_size'

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)

        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # TODO: add the following values and functions
        self.curr_feat = None
        self.update_features(temp_feat)
        self.alpha = 0.1

    # TODO: ReID. for update embeddings during tracking
    def update_features(self, feat):
        
        if(self.curr_feat == None):
            self.curr_feat=feat
        else:
            self.curr_feat = self.curr_feat*(1-self.alpha)+feat*self.alpha
        
        #self.curr_feat=feat

    def activate(self, frame_id):
        """Start a new tracklet"""

        self.track_id = self.next_id()
        self.mean = self._tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean = new_track.tlwh
        if self.score+0.2<new_track.score or new_track.score>0.8:
            self.update_features(new_track.curr_feat)       # TODO: added 20220322
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):     # TODO: 'update_feature' added 02003022
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean = new_tlwh
        self.state = TrackState.Tracked
        self.is_activated = True
        if update_feature or self.score+0.2<new_track.score:                  # TODO: added 20220322
            self.update_features(new_track.curr_feat)
            self.score = new_track.score
        

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        return self.mean
        

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, params=[-100,0.9,100,-100,-100],frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.2
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size


        # [hgx 0119], to see contribution of each module in trakcing
        self.IoU_matched_percent = []
        self.LowScore_matched_percent = []
        self.num1 = 0
        self.num2 = 0
        self.num3 = 0
        self.num4 = 0
        self.params=params
    def update(self, output_results, img_info, img_size, id_feature=None):
        """
        update tracks, e.g. activated, refind, lost and removed tracks
        Args:
            output_results: tensor of shape [bbox_num, 7], 7 for bbox(4) + obj_conf + cls_conf + cls
            img_info:list, [origin_H, origin_W, 1, 1 img_path]
            img_size: tuple, (input_H, input_W)

        Returns:

        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:       # goes here
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]        # score = obj_conf * cls_conf
            bboxes = output_results[:, :4]                              # x1y1x2y2, e.g. tlbr
            id_feature = torch.tensor(output_results[:,7:])
        img_h, img_w = img_info[0], img_info[1]         # origin image size
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))     # input/origin, <1
        bboxes /= scale     # map bbox to origin image size
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh        
        inds_second = np.logical_and(inds_low, inds_high)       # self.args.track_thresh > score > 0.1, for second matching
        dets_second = bboxes[inds_second]                       # detections for second matching
        dets = bboxes[remain_inds]                              # detections for first matching, high quality
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        id_feature_keep = id_feature[remain_inds]
        id_feature_second = id_feature[inds_second]

        # detections for first matching (high score)
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 60) for
                          # detections to STracks, list [OT_0_(0-0), ...]
                          (tlbr, s, f) in
                          zip(dets, scores_keep, id_feature_keep)]       # class STrack in yolox/tracker/byte_tracker.py
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)       # tracked but bot confirmed, e.g. newly init tracks
            else:
                tracked_stracks.append(track)   # normally tracked objects, e.g. activated

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)     # combine tracked_stracks and lost_stracks to strack_pool

        if(len(strack_pool)>0 and len(detections)>0):
            motion = match2(strack_pool,id_feature_keep,pd=self.params[0])
            s1 = match3(strack_pool,id_feature_keep,pd=self.params[0])
            dists = matching.iou_distance(strack_pool, detections)      # IoU distance. 1 - _ious
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)          # refine 'dists' with detection scores
            dists = 1-(1-dists)*np.array(s1)
            #dists = 1-((1-dists)*0.4+0.6*np.array(s1))
            if dists.shape == (1, 1) and motion.shape == (1, 1):
                if motion[0, 0]:
                    #a=1
                    dists[0, 0] = 1
            else:
                #a=1
                dists[motion] = 1
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])    # Hungarain
            self.num1+=len(matches)
            #print(1-dists[matches[:,0],matches[:,1]])
        else:
            dists = matching.iou_distance(strack_pool, detections)      # IoU distance. 1 - _ious
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)          # refine 'dists' with detection scores
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])    # Hungarain
            self.num1+=len(matches)
        for itracked, idet in matches:          # do update w.r.t matching results
            track = strack_pool[itracked]       # track w.r.t index
            det = detections[idet]              # detection w.r.t index
            if track.state == TrackState.Tracked:       # normally tracked successfully tracked
                track.update(detections[idet], self.frame_id,detections[idet].score>0.8)   # update of class STrack, update Kalman Filter, track state and other settings
                activated_starcks.append(track)
            else:                                       # lost tracklets re-found
                track.re_activate(det, self.frame_id, new_id=False)     # re-activate and update Kalman Filter
                refind_stracks.append(track)
        '''Step 2: First association, with high score detection boxes'''
        if(len(matches)>0):
            already_matched = matches[:,1]
            mask = np.ones(scores_keep.shape)
            mask[already_matched] = 0
            id_feature_keep = id_feature_keep[mask==True,:]
            scores_keep = scores_keep[mask==True]
            dets = dets[mask==True,:]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track]


        '''Step 2: First association, with high score detection boxes'''
        #detections = [detections[i] for i in u_detection]#未被匹配的high score框
        #r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]# if r_tracked_stracks[i].state == TrackState.Tracked]#未被匹配的先前关注的框
        if(len(r_tracked_stracks)>0 and len(detections)>0):
            motion = match2(r_tracked_stracks,id_feature_keep,pd=self.params[3])
            dists = matching.iou_distance(r_tracked_stracks, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            if dists.shape == (1, 1) and motion.shape == (1, 1):
                if motion[0, 0]:
                    #a=1
                    dists[0, 0] = 1
            else:
                #a=1
                dists[motion] = 1
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
            self.num3+=len(matches)
        else:
            dists = matching.iou_distance(r_tracked_stracks, detections)  # IoU distance. 1 - _ious
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)  # refine 'dists' with detection scores
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # Hungarain #用IOU又做了一次high score框的match
            self.num3+=len(matches)
        #self.num1+=len(matches)
        for itracked, idet in matches:  # do update w.r.t matching results
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet],
                             self.frame_id,detections[idet].score>0.8)  # update of class STrack, update Kalman Filter, track state and other settings
                activated_starcks.append(track)#high score匹配的加入activated_starcks
            else:
                track.re_activate(det, self.frame_id, new_id=False)#先前因为lost未被激活的框
                refind_stracks.append(track)


        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:  # TODO: 'f, 30' for ReID #low score框
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30) for  # detections_second  --> STrack
                                 (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)]
        else:
            detections_second = []
        second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if
                                  r_tracked_stracks[i].state == TrackState.Tracked]  # TODO: why Tracked?
        
        if(len(second_tracked_stracks)>0 and len(detections_second)>0):
            motion = match2(second_tracked_stracks,id_feature_second,pd=self.params[4])
            dists = matching.iou_distance(second_tracked_stracks, detections_second)      # IoU distance. 1 - _ious
            if dists.shape == (1, 1) and motion.shape == (1, 1):
                if motion[0, 0]:
                    dists[0, 0] = 1
            else:
                dists[motion] = 1
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.45)    # Hungarain
            self.num4+=len(matches)
        else:
            dists = matching.iou_distance(second_tracked_stracks, detections_second)      # IoU distance. 1 - _ious
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.45)    # Hungarain
            self.num4+=len(matches)
        
        for itracked, idet in matches:  # do update w.r.t second matching results
            track = second_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,False)
                activated_starcks.append(track)#low score匹配的加入activated_starcks
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)#先前因为lost未被激活的框

        for it in u_track:  # set unmatched tracks as 'lost'
            track = second_tracked_stracks[it]
            if not track.state == TrackState.Lost:#如果是lost的框，继续lost；如果不是lost框，变成lost
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]  # unmatched detections
        dists = matching.iou_distance(unconfirmed, detections)  # IoU distance. 1 - _ious
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)  # refine 'dists' with detection scores
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id,detections[idet].score>0.8)
            activated_starcks.append(unconfirmed[itracked])#unconfirmed的tracker和det匹配，同第一次association
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)#如果只出现了一帧，就直接扔掉

        """ Step 4: Init new stracks"""
        for inew in u_detection:#剩下的high score框->unconfirmed
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)#unconfirmed的也放进activate
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)#把超过30帧的lost track放入remove

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)#激活的框
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)#重新被激活的框
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)#lost track中减去被激活的
        self.lost_stracks.extend(lost_stracks)#lost track加入新lost的
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)#lost track减去remove的
        self.removed_stracks.extend(removed_stracks)#remove track加上新增的remove的
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)#去重？
        # get scores of lost tracks 
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #self.num2+=len(output_stracks)
        #print(self.num1,self.num2,self.num1/self.num2)
        return output_stracks

def match2(strack_pool,id_feature,pd=-100):
    track_list = []
    tl = []
    for i in strack_pool:
        track_list.append((i.curr_feat).view(1,-1))
        tl.append(i.tlwh)
    E = torch.cat(track_list,0)
    F = id_feature.permute(1,0)
    M = torch.div(E@F,torch.linalg.norm(E,dim=1,keepdim=True)@torch.linalg.norm(F,dim=0,keepdim=True))
    output = M<pd###########################
    return output
def match3(strack_pool,id_feature,pd=-100):
    track_list = []
    tl = []
    for i in strack_pool:
        track_list.append((i.curr_feat).view(1,-1))
        tl.append(i.tlwh)
    E = torch.cat(track_list,0)
    F = id_feature.permute(1,0)
    M = torch.div(E@F,torch.linalg.norm(E,dim=1,keepdim=True)@torch.linalg.norm(F,dim=0,keepdim=True))
    
    return M

  



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

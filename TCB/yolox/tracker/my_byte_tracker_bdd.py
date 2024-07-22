import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from cython_bbox import bbox_overlaps as bboxes_iou
#from yolox.utils import bboxes_iou

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=60,class_id=-1):  # todo: ReID. add inputs of 'temp_feat', 'buffer_size'

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # TODO: add the following values and functions
        self.curr_feat = None
        self.update_features(temp_feat)
        self.alpha = 0.9
        self.class_id = class_id

    # TODO: ReID. for update embeddings during tracking
    def update_features(self, feat):
        
        if(self.curr_feat == None):
            self.curr_feat=feat
        else:
            self.curr_feat = self.curr_feat*(1-self.alpha)+feat*self.alpha
        
        #self.curr_feat=feat

    def predict(self):
        return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        return 
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        #self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean = self._tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        #self.mean, self.covariance = self.kalman_filter.update(
        #    self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        #)
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
        #self.mean, self.covariance = self.kalman_filter.update(
        #    self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
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
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
        

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
    def __init__(self, args, params=[-100,0.9,-100],frame_rate=30):
        self.tracked_stracks_sp = {}
        self.lost_stracks_sp = {}
        self.removed_stracks_sp = {}
        for i in range(1,9):
            self.tracked_stracks_sp[i] = []
            self.lost_stracks_sp[i] = []
            self.removed_stracks_sp[i] = []

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.track_thresh = np.array([0.5,0.5,0.5,0.5,0.5,0.1,0.5,0.5])
        #self.det_thresh =   np.array([0.6,0.6,0.6,0.6,0.6,0.2,0.6,0.6])
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

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
        output_stracks = []
        output_results = output_results.cpu().numpy()
        for i_class in range(1,9):
        #for i_class in range(3,4):
            activated_stracks = []
            refind_stracks = []
            lost_stracks = []
            removed_stracks = []


            per_mask = output_results[:,6]==(i_class-1)
            per_output_results = output_results[per_mask]
            scores = per_output_results[:,4]*per_output_results[:,5]
            bboxes = per_output_results[:,:4]
            id_feature = torch.tensor(per_output_results[:,7:])
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale
            remain_inds = scores > self.track_thresh[i_class-1]
            inds_low = scores > 0.1
            inds_high = scores < self.track_thresh[i_class-1]      
            inds_second = np.logical_and(inds_low, inds_high)       # self.args.track_thresh > score > 0.1, for second matching
            dets_second = bboxes[inds_second]                       # detections for second matching
            dets = bboxes[remain_inds]                              # detections for first matching, high quality
            scores_keep = scores[remain_inds]
            scores_second = scores[inds_second]
            id_feature_keep = id_feature[remain_inds]
            id_feature_second = id_feature[inds_second]
            
            if len(dets) > 0:
                '''Detections'''
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 60, class_id=i_class) for
                              # detections to STracks, list [OT_0_(0-0), ...]
                              (tlbr, s, f) in
                              zip(dets, scores_keep, id_feature_keep)]       # class STrack in yolox/tracker/byte_tracker.py
            else:
                detections = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed = []
            tracked_stracks = []
            for track in self.tracked_stracks_sp[i_class]:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)
            #strack_pool = joint_stracks(tracked_stracks,self.lost_stracks_sp[i_class])
            strack_pool = tracked_stracks
            STrack.multi_predict(strack_pool)
            if(len(strack_pool)>0 and len(detections)>0):
                motion = match2(strack_pool,id_feature_keep,pd=self.params[0])
                s1 = match3(strack_pool,id_feature_keep,pd=self.params[0])
                dists = matching.iou_distance(strack_pool, detections)
                dists = matching.fuse_score(dists, detections)
                dists = 1-(1-dists)*np.array(s1)
                #dists = 1-(1-dists)*0.1+np.array(s1)*0.9
                if dists.shape == (1, 1) and motion.shape == (1, 1):
                    if motion[0, 0]:
                        dists[0, 0] = 1
                else:
                    dists[motion] = 1
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])
                self.num1+=len(matches)
            else:
                dists = matching.iou_distance(strack_pool, detections)
                dists = matching.fuse_score(dists, detections)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])    # Hungarain
                self.num1+=len(matches)


            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id,detections[idet].score>0.7)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            if(len(matches)>0):
                already_matched = matches[:,1]
                mask = np.ones(scores_keep.shape)
                mask[already_matched] = 0
                id_feature_keep = id_feature_keep[mask==True,:]
                scores_keep = scores_keep[mask==True]
                dets = dets[mask==True,:]
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool[i] for i in u_track]

            r_tracked_stracks = joint_stracks(r_tracked_stracks,self.lost_stracks_sp[i_class])
            '''Step 2: First association, with high score detection boxes'''
            if(len(r_tracked_stracks)>0 and len(detections)>0):
                dists = matching.iou_distance(r_tracked_stracks, detections)
                dists = matching.fuse_score(dists, detections)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
                self.num3+=len(matches)
            else:
                dists = matching.iou_distance(r_tracked_stracks, detections)
                dists = matching.fuse_score(dists, detections)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
                self.num3+=len(matches)
            for itracked, idet in matches:  # do update w.r.t matching results
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet],
                                 self.frame_id,detections[idet].score>0.7)  # update of class STrack, update Kalman Filter, track state and other settings
                    activated_stracks.append(track)#high score匹配的加入activated_starcks
                else:
                    track.re_activate(det, self.frame_id, new_id=False)#先前因为lost未被激活的框
                    refind_stracks.append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            if len(dets_second) > 0:  # TODO: 'f, 30' for ReID #low score框
                '''Detections'''
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30,class_id=i_class) for  # detections_second  --> STrack
                                     (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)]
            else:
                detections_second = []
            second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if
                                      r_tracked_stracks[i].state == TrackState.Tracked]
            
            if(len(second_tracked_stracks)>0 and len(detections_second)>0):
                motion = match2(second_tracked_stracks,id_feature_second,pd=self.params[2])
                dists = matching.iou_distance(second_tracked_stracks, detections_second)
                if dists.shape == (1, 1) and motion.shape == (1, 1):
                    if motion[0, 0]:
                        dists[0, 0] = 1
                else:
                    dists[motion] = 1
                matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
                self.num4+=len(matches)
            else:
                dists = matching.iou_distance(second_tracked_stracks, detections_second)
                matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
                self.num4+=len(matches)
            
            for itracked, idet in matches:  # do update w.r.t second matching results
                track = second_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id,False)
                    activated_stracks.append(track)#low score匹配的加入activated_starcks
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:  # set unmatched tracks as 'lost'
                track = second_tracked_stracks[it]
                if not track.state == TrackState.Lost:#如果是lost的框，继续lost；如果不是lost框，变成lost
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            dists = matching.fuse_score(dists, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id,detections[idet].score>0.7)
                activated_stracks.append(unconfirmed[itracked])#unconfirmed的tracker和det匹配，同第一次association
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            """ Step 4: Init new stracks"""
            for inew in u_detection:#剩下的high score框->unconfirmed
                track = detections[inew]
                if track.score < self.det_thresh[i_class-1]:
                    continue
                track.activate(self.kalman_filter, self.frame_id)#unconfirmed的也放进activate
                activated_stracks.append(track)
            """ Step 5: Update state"""
            for track in self.lost_stracks_sp[i_class]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
                elif track.tlbr[0]<10 or track.tlbr[2]>1280-10:
                    track.mark_removed()
                    removed_stracks.append(track)

            self.tracked_stracks_sp[i_class] = [t for t in self.tracked_stracks_sp[i_class] if t.state == TrackState.Tracked]
            self.tracked_stracks_sp[i_class] = joint_stracks(self.tracked_stracks_sp[i_class], activated_stracks)
            self.tracked_stracks_sp[i_class] = joint_stracks(self.tracked_stracks_sp[i_class], refind_stracks)
            self.lost_stracks_sp[i_class] = sub_stracks(self.lost_stracks_sp[i_class], self.tracked_stracks_sp[i_class])
            self.lost_stracks_sp[i_class].extend(lost_stracks)
            self.lost_stracks_sp[i_class] = sub_stracks(self.lost_stracks_sp[i_class], self.removed_stracks_sp[i_class])
            self.removed_stracks_sp[i_class].extend(removed_stracks)
            self.tracked_stracks_sp[i_class], self.lost_stracks_sp[i_class] = remove_duplicate_stracks(self.tracked_stracks_sp[i_class], self.lost_stracks_sp[i_class])
            output_stracks.extend([track for track in self.tracked_stracks_sp[i_class] if track.is_activated])
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
    output = M<pd#########################################################################################
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
def match1(starck_pool,id_feature,scores,bboxes,param):
    track_list = []
    tl = []
    for i in starck_pool:
        track_list.append((i.curr_feat).view(1,-1))
        tl.append(i.tlwh)
    E = torch.cat(track_list,0)
    F = id_feature.permute(1,0)
    M = torch.div(E@F,torch.linalg.norm(E,dim=1,keepdim=True)@torch.linalg.norm(F,dim=0,keepdim=True))
    __,index = torch.max(M,1)
    selected_track = np.full((F.shape[1],),-1)
    matches=[] 
    u_track=[]
    for i in range(0,len(starck_pool)):
        ratio = (bboxes[index[i]][2]-bboxes[index[i]][0])*(bboxes[index[i]][3]-bboxes[index[i]][1])/(starck_pool[i].tlwh[2]*starck_pool[i].tlwh[3])
        box1 = np.array(starck_pool[i].tlbr).reshape((1,4))
        box2 = np.array(bboxes[index[i]]).reshape((1,4))
        iou = bboxes_iou(np.ascontiguousarray(box1,dtype=np.float),np.ascontiguousarray(box2,dtype=np.float))
        if(scores[index[i]]>0.3 and ratio>0.5 and ratio<2 and iou>param):
            if(__[i]>0.7):
            #if(__[i]>0.95):#find pair
                if(selected_track[index[i]]==-1):# is not matched by previous tracker
                    selected_track[index[i]]=i
                else:
                    if(__[i]>__[selected_track[index[i]]]):
                        selected_track[index[i]]=i
    match_track = []
    for i in range(0,len(selected_track)):
        if(selected_track[i]!=-1):
            match_track.append(selected_track[i])
            matches.append([selected_track[i],i])
    matches = np.asarray(matches)
    for i in range(0,len(starck_pool)):
        if i not in match_track:
            u_track.append(i)
    u_track = np.asarray(u_track)
    return matches,u_track
  



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

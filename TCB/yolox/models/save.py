from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

import math
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
import numpy as np

class YOLOXHead2(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8,16,32],
            in_channels=[256,512,1024],
            act="silu",
            depthwise=False,
            nID=None,
    ):
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.nID = nID+1
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.reid_convs = nn.ModuleList()
        self.reid_preds = nn.ModuleList()
        self.reid_adjust1 = nn.ModuleList()
        self.reid_adjust2 = nn.ModuleList()
        self.emb_dim = 256
        self.reid_classifier = nn.Linear(self.emb_dim, self.nID)
        self.reid_Loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.strides = strides
        #self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        """
        self.alliou = 0
        self.iou07 = 0
        self.iou08 = 0
        self.iou09 = 0
        self.iou05less = 0
        """
        self.dimmatch = [0,0,0]
        self.dimnum = [0,0,0]
        self.dimall = 0
        self.dimfind = [0,0,0]
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):# 表示三个尺度主干网络的输出
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(# cls_convs与reg_convs网络层相同
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(#类别预测
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(#bbox预测
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(#置信度预测
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reid_convs.append(
                nn.Sequential(      # 2 BaseConv layers
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reid_preds.append(      # 1 Conv2d layer, output channel is 'self.emb_dim'
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.emb_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def forward(self,xin, labels=None, imgs=None,last_reid=None,feature_id=None):
        outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            reid_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            reid_feat = self.reid_convs[k](reid_x)
            reid_output = self.reid_preds[k](reid_feat)

            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid(), reid_output],1)
            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        yolo_outputs = self.decode_outputs(outputs,dtype=xin[0].type())#[[b,h1,w1,134],[b,h2,w2,134],[b,h3,w2,w3,134]]
        if not self.training:#test
            if last_reid is None:#test first frame
                yolo_outputs = torch.cat(
                    [x.view(x.shape[0],-1,4+1+self.num_classes+self.emb_dim) for x in yolo_outputs], dim=1
                )
                return yolo_outputs
            else:#test
                return self.reid_check(yolo_outputs, last_reid,xin)
        else:#training
            if(last_reid==None):#previous frame, return reid feature
                return self.get_reid(yolo_outputs, labels)
            else:
                return self.checknet(yolo_outputs, labels, last_reid,xin,xin[0].type(),feature_id)        
            
    def decode_outputs(self, outputs, dtype):
        out = []

        for (hsize, wsize), stride, output in zip(self.hw, self.strides, outputs):
            output = output.flatten(start_dim=2).permute(0,2,1)
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1,-1,2).type(dtype)
            output[...,:2] = ((output[...,:2]+grid)*stride).type(dtype)
            output[...,2:4] = (torch.exp(output[...,2:4])*stride).type(dtype)
            output = output.view(-1,hsize,wsize,4+1+self.num_classes+self.emb_dim)#[batch,h,w,134]
            out.append(output)
        return out

    def cal(self,topk,center,wsize,out):
        for i in topk:
            x = torch.div(i,wsize,rounding_mode="floor")
            y = i%wsize
            if(bboxes_iou((out[x,y,:4]).view(1,4),(out[center[0],center[1],:4]).view(1,4),False)[0][0]>0.8):
                return 1
        return 0            
    def select1(self,topk,out,wsize):
        sel_pos = []
        sel_num = []
        for i in range(len(topk)):
            x = torch.div(topk[i],wsize,rounding_mode="floor")
            y = topk[i]%wsize
            flag=False
            if(out[x,y,4]*out[x,y,5]<0.1):
                continue
            for j in range(len(sel_pos)):
                if(bboxes_iou((out[x,y,:4]).view(1,4),(out[sel_pos[j][0],sel_pos[j][1],:4]).view(1,4),False)[0][0]>0.8):
                #if(abs(x-sel_pos[j][0])+abs(y-sel_pos[j][1])<=2):#similar matching
                    flag=True
                    if(out[x][y][4]*out[x][y][5]>out[sel_pos[j][0]][sel_pos[j][1]][4]*out[sel_pos[j][0]][sel_pos[j][1]][5]):#better
                        sel_pos[j]=(x,y)
                        sel_num[j]=i
                        break
            if(flag==True):
                continue
            sel_pos.append((x,y))
            sel_num.append(i)

        return topk[sel_num]
    
    def checknet(self,yolo_outputs, labels, last_reid,xin,dtype,feature_id):
        torch.cuda.empty_cache()
        reid_loss = 0
        loss = 0
        match1 = 0
        match2 = 0
        dims,dims2 = self.get_dim(yolo_outputs,labels,feature_id)
        for k,((hsize, wsize),stride,output) in enumerate(zip(self.hw,self.strides,yolo_outputs)):#[b,h1,w1,134]
            Ms = []
            Ga = []
            for batch_idx in range(output.shape[0]):#every image
                gau_dim = dims[batch_idx][k]
                if(len(dims2[batch_idx][k])==0):#no last_reid of this dimension
                    continue
                reid_feat = output[batch_idx,...,6:]#[h,w,128]
                E = last_reid[batch_idx][k][dims2[batch_idx][k]].view(-1,self.emb_dim)#[n,128]
                F = reid_feat.view(-1,self.emb_dim).permute(1,0)#[128,h*w]
                M = E@F
                #use cosine distance
                #shape of M: [n,h*w]
                # TODO: get mask
                gaussian = []
                _,index = torch.max(M,1)
                index_x = torch.div(index,wsize,rounding_mode="floor")#index_x = index//wsize
                index_y = index%wsize
                select = output[batch_idx,index_x,index_y,6:]
                idloss = self.reid_classifier(select)
                id_target = (torch.tensor(feature_id[batch_idx][k])[dims2[batch_idx][k]].type(dtype)).to(torch.int64)
                id_match = []
                topk, topk_idx = torch.topk(M,15,dim=1)
                reg_position = output[batch_idx,...,:4].view(-1,4)
                for i in range(len(gau_dim)):
                    #result1 = (M[i].reshape([self.hw[k][0],self.hw[k][1]])).cuda().data.cpu().numpy()
                    #np.savetxt("./heat/h/heat_{}_{}.txt".format(k,i),result1)
                    gt_position = labels[batch_idx,gau_dim[i],1:5].view(1,4)
                    gt_iou = bboxes_iou(gt_position, reg_position,False)
                    _,gt_index = torch.max(gt_iou,1)
                    center_gt = (int(gt_index//wsize),int(gt_index%wsize))
                    #tl_gt = (labels[batch_idx][gau_dim[i]][2],labels[batch_idx][gau_dim[i]][1])
                    #center_gt = [int(torch.div(tl_gt[0],stride,rounding_mode="floor")),int(torch.div(tl_gt[1],stride,rounding_mode="floor"))]
                    #if(center_gt[0]>=hsize):
                    #    center_gt[0]=hsize-1
                    #if(center_gt[1]>=wsize):
                    #    center_gt[1]=wsize-1
                    per_gaussian = self.get_gaussian(center_gt,hsize,wsize)#[hsize,wsize]
                    #np.savetxt("./heat/h/gt_{}_{}.txt".format(k,i),per_gaussian.numpy())
                    gaussian.append(per_gaussian.view(1,hsize*wsize)) 
                    sel = self.select1(topk_idx[i],output[batch_idx],wsize)
                    self.dimfind[k]+=len(sel)
                    out_match= self.cal(sel,center_gt,wsize,output[batch_idx])
                    match1+=out_match
                    match2+=1
                    self.dimnum[k]+=1
                    self.dimmatch[k]+=out_match
                    output[batch_idx,index_x[i],index_y[i],:4].view(1,4)
                    if(bboxes_iou((output[batch_idx,index_x[i],index_y[i],:4]).view(1,4),(output[batch_idx,center_gt[0],center_gt[1],:4]).view(1,4),False)[0][0]>0.8):
                        id_match.append(i)
                id_target = id_target[id_match]
                idloss = idloss[id_match]
                if(len(idloss)>0):
                    reid_loss+=self.reid_Loss(idloss,id_target)

                gaussian = torch.cat(gaussian,dim=0)#[n,h*w]
                Ms.append(M.view(1,E.shape[0],hsize*wsize))
                Ga.append(gaussian.view(1,gaussian.shape[0],gaussian.shape[1]))
            if(len(Ms)==0):
                continue
            Ms_all = torch.cat(Ms)#[batch,n,h1*w1]
            Ga_all = torch.cat(Ga)#[batch,h*w]->[batch,n,h1*w1]
            #Ms_final = (torch.sum(Ms_all,dim=1,keepdim=True)-0.5)*10#[batch,1,h1,w1]
            Ms_final = Ms_all.view(-1,hsize*wsize)#[batch*n,h1*w1]
            Ga_all = Ga_all.view(-1,hsize*wsize).type(dtype)
            Ms_out = Ms_final.sigmoid()

            for i in range(0,Ms_out.shape[0]):
                per_g=Ga_all[i]
                per_m=Ms_out[i]
                positive = per_g>=1
                negative = ~positive
                loss+=-(((1-per_m[positive])*torch.log(per_m[positive]+1e-7)).sum()+
                        ((1-per_g[negative])*per_m[negative]*torch.log(1-per_m[negative]+1e-7)).sum())/(hsize*wsize*E.shape[0]*3)
            """ 
            positive = Ga_all>=1
            negative = ~positive
            loss += -(((1-Ms_out[positive])*torch.log(Ms_out[positive]+1e-7)).sum()+
                     ((1-Ga_all[negative])*Ms_out[negative]*torch.log(1-Ms_out[negative]+1e-7)).sum())/(hsize*wsize*output.shape[0])
            #loss+=self.bcewithlog_loss(Mp,Ga_all).sum()/(hsize*wsize*output.shape[0])
            """
        #print(self.alliou,self.iou07,self.iou08,self.iou09,self.iou05less)
        total_loss = loss+reid_loss
        match_rate = match1/match2
        print("match:{}, find match:{}, match object:{}, all object:{}".format(self.dimmatch,self.dimfind,self.dimnum,self.dimall))
        #print(self.dimmatch,self.dimnum,self.dimall)
        return total_loss,loss,reid_loss, match_rate,match1,match2



    def get_gaussian(self,center,hsize,wsize):
        x_left = -(center[0]-0)
        x_right = hsize-center[0]-1
        y_left = -(center[1]-0)
        y_right = wsize-center[1]-1
            
        yv,xv = torch.meshgrid([torch.arange(x_left,x_right+1),torch.arange(y_left,y_right+1)])
        sigma_square = 0.75*0.75
        g = torch.exp(-((xv**2 + yv**2)/(2*sigma_square)))#[hsize,wsize]
        return g


    def get_reid(self,yolo_outputs, labels):
        torch.cuda.empty_cache()
        out = []
        out2 = []
    
        for batch_idx in range(yolo_outputs[0].shape[0]):
            re = []
            score = []
            for output in yolo_outputs:#three dimensions
                reid_output = output[batch_idx,...,6:]#[H,W,128]
                reid_output = reid_output.view(-1,self.emb_dim)
                reg_output = output[batch_idx,...,:4]#[H,W,4]
                reg_output = reg_output.view(-1,4)
                #obj_output = output[batch_idx,...,4]
                #obj_output = obj_output.view(-1,1)
                nlabel = (labels.sum(dim=2)>0).sum(dim=1)
                
                gt_label = labels[batch_idx,:nlabel,1:5]
                iou = bboxes_iou(gt_label,reg_output, False)
                #_,index = torch.max(iou*obj_cls_output[:,0]*obj_cls_output[:,1],1)
                _,index = torch.max(iou,1)


                reid_feat = reid_output[index]#[nlabel,128]
                re.append(reid_feat.detach())
                score.append(_)
            scores = torch.cat(score, dim=0).view(3,-1)
            _,chooseindex = torch.max(scores,0)
            
            out_per_batch = [[],[],[]]
            out2_per_batch = [[],[],[]]
            for i in range(0,3):
                for j in range(0,len(scores[0])):
                    if(scores[i][j]>0.8):#threshold:iou>0.65
                        out_per_batch[i].append(re[chooseindex[j]][j])
                        out2_per_batch[i].append(labels[batch_idx,j,5])
            if(len(out_per_batch[0])>0):
                out_per_batch[0] = torch.cat(out_per_batch[0],dim=0).view(-1,self.emb_dim)
            if(len(out_per_batch[1])>0):
                out_per_batch[1] = torch.cat(out_per_batch[1],dim=0).view(-1,self.emb_dim)
            if(len(out_per_batch[2])>0):
                out_per_batch[2] = torch.cat(out_per_batch[2],dim=0).view(-1,self.emb_dim)
            out.append(out_per_batch)
            out2.append(out2_per_batch)
        #out = torch.vstack(out)
        return out,out2
    def get_dim(self,yolo_outputs, labels,feature_id):
        out = []
        out2 = []
        self.dimall+=(labels.sum(dim=2)>0).sum(dim=1)
        for batch_idx in range(yolo_outputs[0].shape[0]):
            score = []
            for output in yolo_outputs:#three dimensions

                reg_output = output[batch_idx,...,:4]#[H,W,4]
                reg_output = reg_output.view(-1,4)
                #obj_output = output[batch_idx,...,4]
                #obj_output = obj_output.view(-1,1)
                nlabel = (labels.sum(dim=2)>0).sum(dim=1)
                
                
                gt_label = labels[batch_idx,:nlabel,1:5]
                iou = bboxes_iou(gt_label,reg_output, False)
                _,index = torch.max(iou,1)

                #scores_feat = obj_output[index]#[nlabel,2]
                score.append(_)
            scores = torch.cat(score, dim=0).view(3,-1)
            #_,chooseindex = torch.max(scores,0)
            
            out_per_batch = [[],[],[]]
            out2_per_batch = [[],[],[]]
            feat = [[],[],[]]
            for i in range(0,3):
                for j in range(0,len(scores[0])):
                    if(scores[i][j]>0.8):
                        track_id=labels[batch_idx,j,5]
                        if(track_id in feature_id[batch_idx][i]):
                            out_per_batch[i].append(j)
                            feat[i].append(track_id)
            for i in range(0,3):
                for j in range(0,len(feature_id[batch_idx][i])):
                    if(feature_id[batch_idx][i][j] in feat[i]):
                        out2_per_batch[i].append(j)
            out.append(out_per_batch)
            out2.append(out2_per_batch)
        #out = torch.vstack(out)
        return out,out2

    def reid_check(self,yolo_outputs, last_reid,xin):
        #batch_size=1
        for (hsize, wsize),stride,output,adjust1,adjust2,x in zip(self.hw,self.strides,yolo_outputs,self.reid_adjust1,self.reid_adjust2,xin):#[b,h1,w1,134]
            Ms = []
            for batch_idx in range(output.shape[0]):#every image
                reid_feat = output[batch_idx,...,6:]#[h,w,128]
                E = last_reid[batch_idx].view(-1,self.emb_dim)#[n,128]
                F = reid_feat.view(-1,self.emb_dim).permute(1,0)#[128,h*w]
                M = torch.div(E@F,torch.linalg.norm(E,dim=1,keepdim=True)@torch.linalg.norm(F,dim=0,keepdim=True))
                #use cosine distance
                #shape of M: [n,h*w]
                # TODO: get mask
                r = 2
                mask = []
                _,index = torch.max(M,1)
                index_x = torch.div(index,wsize,rounding_mode="floor")#index_x = index//wsize
                index_y = index%wsize
                for i in range(0,E.shape[0]):
                    center = (index_x[i],index_y[i])#coordinate of the max respond
                    per_mask = torch.zeros([hsize,wsize])
                    tl_coor = (max(0,center[0]-r),max(0,center[1]-r))
                    br_coor = (min(hsize-1,center[0]+r),min(wsize-1,center[1]+r))
                    per_mask[tl_coor[0]:br_coor[0]+1, tl_coor[1]:br_coor[1]+1] = 1

                    mask.append(per_mask.view(1,hsize*wsize))
                mask = torch.cat(mask,dim=0)#[n,h*w]
                Ms.append((M*mask).view(1,E.shape[0],hsize,wsize))
            Ms_all = torch.cat(Ms)#[batch,n,h1,w1]
            Ms_final = torch.sum(Ms_all,dim=1,keepdim=True)#[batch,1,h1,w1]
            Ms_adjust1 = adjust1(Ms_final)#[batch,1,h1,w1]
            F_adjust = Ms_adjust1*x#[batch,320/640/1280,h1,w1]
            Mp = adjust2(F_adjust)#[batch,1,h1,w1]
            #now I get: Mp[batch,1,h,w]
            Mp = Mp.view(output.shape[0],hsize,wsize).sigmoid()#某维度下的特征响应图
            #yolo_outputs:#[[b,h1,w1,134],[b,h2,w2,134],[b,h3,w2,w3,134]]
            selected = Mp>0.7#set threshold,[batch,h,w]
            #found the coordinate that need to be updated
            output[selected][4] = output[selected][4]/2+0.45# need more research

        return yolo_outputs
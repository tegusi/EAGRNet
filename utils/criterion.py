import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
   
    def parsing_loss_bk(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        target[1] = torch.clamp(target[1], 0, 1)
        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)

        return loss
        
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)
        loss = 0

        # loss for parsing
        preds_parsing = preds
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target)
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target)


        return loss
    def forward(self, preds, target):  
        loss = self.parsing_loss_bk(preds, target) 
        return loss
    
class CriterionCrossEntropyEdgeParsing_boundary_eagrnet_loss(nn.Module):
    """Weighted CE2P loss for face parsing.
    
    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """
    def __init__(self, loss_weight=[1.0, 1.0, 1.0], ignore_index=255, num_classes=11):
        super(CriterionCrossEntropyEdgeParsing_boundary_attention_loss, self).__init__()
        self.ignore_index = ignore_index   
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index) 
        self.loss_weight = loss_weight
          
    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        
        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = torch.from_numpy(np.array(weights)).float().cuda()

        edge_p_num = target[1].cpu().numpy().reshape(target[1].size(0),-1).sum(axis=1)
        edge_p_num = np.tile(edge_p_num, [h, w, 1]).transpose(2,1,0)
        edge_p_num = torch.from_numpy(edge_p_num).cuda().float()
        
        loss_edge = 0; loss_parse = 0; loss_att_parse = 0; loss_att_edge = 0
        for i in range(len(preds)):
            preds_ = preds[i]
            scale_parse = F.upsample(input=preds[0], size=(h, w), mode='bilinear') # parsing
            scale_edge = F.upsample(input=preds[1], size=(h, w), mode='bilinear')  # edge 

            loss_parse_ = self.criterion(scale_parse, target[0])
            loss_edge_ = F.cross_entropy(scale_edge, target[1], weights)
            loss_att_edge_ = self.criterion_weight(scale_parse, target[0]) * target[1].float()
            loss_att_edge_ = loss_att_edge_ / edge_p_num  # only compute the edge pixels
            loss_att_edge_ = torch.sum(loss_att_edge_) / target[1].size(0)  # mean for batchsize      
                  
            loss_parse += loss_parse_
            loss_edge += loss_edge_
            loss_att_edge += loss_att_edge_
        
        # print('loss_parse: {}\t loss_edge: {}\t loss_att_edge: {}'.format(loss_parse,loss_edge,loss_att_edge))
        return self.loss_weight[0]*loss_parse + self.loss_weight[1]*loss_edge + self.loss_weight[2]*loss_att_edge
        
class CriterionCrossEntropyEdgeParsing_boundary_agrnet_loss(nn.Module):
    """Weighted CE2P loss for face parsing.
    
    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """
    def __init__(self, loss_weight=[1.0, 1.0, 1.0], ignore_index=255, num_classes=11):
        super(CriterionCrossEntropyEdgeParsing_boundary_agrnet_loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index   
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index) 
        self.loss_weight = loss_weight
          
    def calculate_distance_term(self, means, n_objects, delta_d, norm=2, usegpu=True):
        """means: bs, n_instances, n_filters"""

        bs, n_instances, n_filters = means.size()

        dist_term = 0.0
        for i in range(bs):
            _n_objects_sample = n_objects

            if _n_objects_sample <= 1:
                continue

            _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
            means_1 = _mean_sample.unsqueeze(1).expand(
                _n_objects_sample, _n_objects_sample, n_filters)
            means_2 = means_1.permute(1, 0, 2)

            diff = means_1 - means_2  # n_objects, n_objects, n_filters

            _norm = torch.norm(diff, norm, 2)

            margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
            margin = margin.to(_norm.device)

            _dist_term_sample = torch.sum(
                torch.clamp(margin - _norm, min=0.0) ** 2)
            _dist_term_sample = _dist_term_sample / \
                (_n_objects_sample * (_n_objects_sample - 1))
            dist_term += _dist_term_sample

        dist_term = dist_term / bs

        return dist_term


    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        
        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = torch.from_numpy(np.array(weights)).float().cuda()

        edge_p_num = target[1].cpu().numpy().reshape(target[1].size(0),-1).sum(axis=1)
        edge_p_num = np.tile(edge_p_num, [h, w, 1]).transpose(2,1,0)
        edge_p_num = torch.from_numpy(edge_p_num).cuda().float()
        
        loss_edge = 0; loss_parse = 0; loss_att_parse = 0; loss_att_edge = 0
        scale_parse = F.upsample(input=preds[0], size=(h, w), mode='bilinear') # parsing
        scale_edge = F.upsample(input=preds[1], size=(h, w), mode='bilinear')  # edge 
        scale_proj = F.upsample(input=preds[3], size=(h, w), mode='bilinear')  # edge 

        loss_parse = self.criterion(scale_parse, target[0])
        loss_proj = self.criterion(scale_proj, target[0])
        loss_edge = F.cross_entropy(scale_edge, target[1], weights)
        loss_anchor = self.calculate_distance_term(preds[2], self.num_classes, 1)
        loss_att_edge_ = self.criterion_weight(scale_parse, target[0]) * target[1].float()
        loss_att_edge_ = loss_att_edge_ / (edge_p_num + 1)  # only compute the edge pixels
        loss_att_edge = torch.sum(loss_att_edge_) / target[1].size(0)  # mean for batchsize      
                
        # loss_parse += loss_parse_ + loss_anchor
        # loss_edge += loss_edge_
        # loss_att_edge += loss_att_edge_
        
        # print('loss_parse: {}\t loss_edge: {}\t loss_att_edge: {}\t loss_anchor: {}'.format(loss_parse,loss_edge,loss_att_edge,loss_anchor))
        return self.loss_weight[0]*(loss_parse + loss_proj) + self.loss_weight[1]*loss_edge + self.loss_weight[2]*loss_att_edge + self.loss_weight[3] * loss_anchor
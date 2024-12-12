# @Time    : 2024/06/01 21:09
# @Author  : Yang Xionghui
# @File    : loss.py
# @Software: PyCharm

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:03:06 2023

@author: Sana
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class FocalLoss(nn.Module):
#     """
#     参考 https://github.com/lonePatient/TorchBlocks
#     """

#     def __init__(self, gamma=2.0, alpha=[0.2, 0.2, 0.2, 0.2, 0.2], epsilon=1.e-9, device=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha, device=device)
#         else:
#             self.alpha = alpha
#         self.epsilon = epsilon

#     def forward(self, input, target):
#         """
#         Args:
#             input: model's output, shape of [batch_size, num_cls]
#             target: ground truth labels, shape of [batch_size]
#         Returns:
#             shape of [batch_size]
#         """
#         num_labels = input.size(-1)
#         idx = target.view(-1, 1).long()
#         one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         one_hot_key[:, 0] = 0  # ignore 0 index.
#         logits = torch.softmax(input, dim=-1)
#         loss = -self.alpha.to(target.device) * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
#         loss = loss.sum(1)
#         return loss.mean()

#alpha=[0.61, 0.82, 0.73,0.87,0.97]
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.0006,0.0013,0.0008,0.0018,0.008], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha.to(target.device)[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
class FocalLoss(nn.Module):
    
    def __init__(self, alpha=[1, 1.1, 1, 1.05, 1.05], gamma=1):
        super().__init__()
        self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
    
    def forward(self, output, target):
        # print(output.shape)
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha.to(target.device))
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list=[2295,1051,1504,752,175], max_m=0.2, weight=[1, 1.1, 1, 1.05, 1.05], s=1):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
   
        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # m_list = torch.cuda.FloatTensor(m_list)
        m_list = torch.tensor(m_list, dtype=torch.float32, device='cuda')
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = torch.tensor(weight, dtype=torch.float32, device='cuda')
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        
        index_bool = index.bool()
        output = torch.where(index_bool, x_m, x)
        # output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class LMFLoss(nn.Module):
        def __init__(self,cls_num_list=[1595,740,1115,555,125],weight=[1, 1.1, 1.05, 1.05, 1.05],alpha=1,beta=1, gamma=1, max_m=0.2, s=1):
            super().__init__()
            self.focal_loss = FocalLoss(weight, gamma)
            self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
            # self.focal_loss = FocalLoss()
            # self.ldam_loss = LDAMLoss()
            self.alpha= alpha
            self.beta = beta

        def forward(self, output, target):
            focal_loss_output = self.focal_loss(output, target)
            ldam_loss_output = self.ldam_loss(output, target)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss  
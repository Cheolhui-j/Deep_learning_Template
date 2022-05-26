import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import sys
sys.path.append('../')
from config import cfg

class magface(nn.Module):

    def __init__(self, cfg, num_class=10):
        super(magface, self).__init__()
        
        self.scale = cfg.loss_s
        self.emb_size = cfg.emd_size
        self.num_cls = num_class

        self.u_a = cfg.u_a
        self.l_a = cfg.l_a

        self.u_m = cfg.u_m
        self.l_m = cfg.l_m

        self.easy_margin = cfg.easy_margin

        # weight matrix init
        self.weight_matrix = torch.nn.Parameter(torch.empty(self.num_cls, self.emb_size))
        nn.init.xavier_normal_(self.weight_matrix)

        # self.criterion = nn.CrossEntropyLoss()

    def _calc_m(self, embedding_norm):
        m_a = (self.u_m-self.l_m)/(self.u_a - self.l_a) * (embedding_norm - self.l_a) + self.l_m 
        return m_a

    def _calc_g(self, embedding_norm):
        g_a = 1 / embedding_norm + 1 / (self.u_a ** 2) * embedding_norm
        return torch.mean(g_a)

    def _calc_lambda(self):
        lambda_g = ((self.scale*(self.u_a**2)*(self.l_a**2)) / ((self.u_a**2)-(self.l_a**2))) * (self.u_m-self.l_m)/(self.u_a - self.l_a)
        return lambda_g

    def forward_old(self, embedding, label):

        # 1. x, w normalize & 2. calculate x * w = similarity
        sim = nn.functional.linear(nn.functional.normalize(embedding ,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight_matrix,p=2, dim=1, eps=1e-12)).clamp(-1, 1)

        # only magface
        embedding_norm = torch.norm(embedding, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m_a = self._calc_m(embedding_norm)
        g_a = self._calc_g(embedding_norm)
        lambda_g = self._calc_lambda() 

        # 3. original_target_logit
        # cos = sim[:, label]
        gt_label = label.unsqueeze(axis=-1)
        cos = torch.gather(sim, 1, gt_label)
        # cos = sim

        # 4. calculate theta
        theta = torch.acos(cos)

        # 5. marginal_target_logit
        cos_add_m = torch.cos(theta + m_a)

        # 6. one_hot encoding 
        index = F.one_hot(label, num_classes = self.num_cls).type(dtype = torch.bool)

        # 7. sim 
        # # add easy_margin 
        if self.easy_margin:
            cos_add_m = torch.where(
                sim > 0, cos_add_m, theta)
        else:
            th = torch.cos(torch.tensor(math.pi) - m_a)
            sinmm = torch.sin(torch.tensor(math.pi) - m_a) * m_a
            cos_add_m = torch.where(
                sim > th, cos_add_m, theta - sinmm)

        # cos_add_m, cos = self.scale * cos_add_m, self.scale * cos 

        sim = sim + index * (cos_add_m - cos)

        # 8. s * sim
        sim = sim * self.scale

        # loss = self.criterion(sim, label)
        loss = F.cross_entropy(sim, label, reduction='none')

        return loss.mean() + lambda_g * g_a 

    def forward(self, embedding, label):

        # 1. x, w normalize & 2. calculate x * w = similarity
        sim = nn.functional.linear(nn.functional.normalize(embedding ,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight_matrix,p=2, dim=1, eps=1e-12)).clamp(-1, 1)

        # only magface
        embedding_norm = torch.norm(embedding, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m_a = self._calc_m(embedding_norm)
        cos_m, sin_m = torch.cos(m_a), torch.sin(m_a)

        g_a = self._calc_g(embedding_norm)
        lambda_g = self._calc_lambda() 

        cos = sim 
        sin = torch.sqrt(1.0 - torch.pow(sim, 2))
        cos_m = cos * cos_m - sin * sin_m

        if self.easy_margin:
            cos_m = torch.where(
                cos > 0, cos_m, cos)
        else:
            th = torch.cos(torch.tensor(math.pi) - m_a)
            sinmm = torch.sin(torch.tensor(math.pi) - m_a) * m_a
            cos_m = torch.where(
                cos > th, cos_m, cos - sinmm)

        cos, cos_m = cos * self.scale, cos_m * self.scale

        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = one_hot * cos_m + (1.0 - one_hot) * cos

        loss = F.cross_entropy(output, label, reduction='mean')

        return loss.mean() + lambda_g * g_a 
        

if __name__ == "__main__":

    emb = torch.rand(64, 512)
    label = torch.randint(high=9, size=(64,))

    # emb = torch.empty(64, 32).fill_(0.01)
    # label = torch.empty(64).fill_(1).type(torch.long)

    loss = magface(cfg)

    out = loss(emb, label)

    print(out)




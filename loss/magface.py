import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class magface(nn.Module):

    def __init__(self, scale = 80, emb_size = 32, num_class = 10):
        super(magface, self).__init__()
        
        self.scale = scale
        self.emb_size = emb_size
        self.num_cls = num_class

        self.u_a = 110
        self.l_a = 10

        self.u_m = 0.8
        self.l_m = 0.4

        self.easy_margin = True

        # weight matrix init
        self.weight_matrix = torch.nn.Parameter(torch.empty(self.num_cls, self.emb_size))
        nn.init.xavier_normal_(self.weight_matrix)

        # self.criterion = nn.CrossEntropyLoss()

    def _calc_lambda(self):
        lambda_g = ((self.scale*(self.u_a**2)*(self.l_a**2)) / ((self.u_a**2)-(self.l_a**2))) * (self.u_m-self.l_m)/(self.u_a - self.l_a)
        return lambda_g

    def _calc_g(self, embedding_norm):
        g_a = 1 / embedding_norm + 1 / (self.u_a ** 2) * embedding_norm
        return torch.mean(g_a)

    def forward(self, embedding, label):

        # 1. x, w normalize & 2. calculate x * w = similarity
        sim = nn.functional.linear(nn.functional.normalize(embedding ,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight_matrix,p=2, dim=1, eps=1e-12))

        # only magface
        embedding_norm = torch.norm(embedding, dim=1, keepdim=True)
        m_a = (self.u_m-self.l_m)/(self.u_a - self.l_a) * (embedding_norm - self.l_a) + self.l_m 
        g_a = self._calc_g(embedding_norm)
        lambda_g = self._calc_lambda() 

        # 3. original_target_logit
        # cos = sim[:, label]
        gt_label = label.unsqueeze(axis=-1)
        cos = torch.gather(sim, 1, gt_label)

        # 4. calculate theta
        theta = torch.acos(cos)

        # 5. marginal_target_logit
        cos_add_m = torch.cos(theta + m_a)

        # 6. one_hot encoding 
        index = F.one_hot(label, num_classes = self.num_cls).type(dtype = torch.bool)

        # 7. sim 
        # sim = sim + index * (cos_add_m - cos)
        if self.easy_margin:
            final_sim = torch.where(
                sim > 0, cos_add_m, theta)
        else:
            th = math.cos(math.pi - m_a)
            sinmm = math.sin(math.pi - m_a) * m_a
            final_sim = torch.where(
                sim > th, cos_add_m, theta - sinmm)

        # 8. s * sim
        final_sim = final_sim * self.scale

        # loss = self.criterion(sim, label)
        loss = F.cross_entropy(sim, label, reduction='none')

        return loss.mean() + lambda_g * g_a

if __name__ == "__main__":

    emb = torch.rand(64, 32)
    label = torch.randint(high=9, size=(64,))

    loss = magface()

    print(loss(emb, label))
    


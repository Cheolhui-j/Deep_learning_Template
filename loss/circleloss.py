import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 왜 내가 짠 코드로 하면 loss가 잘 안 주는지 모르겠으므로
# 비교바람. --> 02.21

# 02.22?
# 원인 찾음
# 총 2개의 이유가 있었음.

# 1. one_hot 코딩할 때, 값들을 boolean값으로 변환해주지 않을 경우
# 0, 1로 구성이 되는데 이렇게 되면 1인 index만을 선택하는 게 아닌 0,1 전부 다 선택해서 
# p, n의 값을 계산할 때 필요없는 값들이 고정되므로 loss 값이 줄어들지 않음.

# 2. one_hot 코딩이 되었더라도 49줄과 같이 곱해주는 경우에는 위의 문제들이
# 동일하게 발생하고 똑같이 값이 고정되어 loss값이 줄지 않음.

# 정리하자면,  원하는 형태는 class가 10개일 경우
# p = (64, 1) 의 참인 index의 값으로 이루어진 배열이어야 하고
# n = (64, 9) 의 나머지 거짓인  index의 값으로 이루어진 배열이어야 함.

# 허나 위의 형태로 선언하게 되면 p, n 모두 (64, 10)으로 계산되어 
# 불필요한 index까지 포함하게 되어 값이 고정되는 현상.

# Notion에 정리 바람. 예시 포함해서.
 
class circle_loss(nn.Module):

    def __init__(self, gamma = 80, m = 0.25, emb_size = 32, num_class = 10):
        super(circle_loss, self).__init__()
        
        self.gamma = gamma
        self.m = m
        self.emb_size = emb_size
        self.num_cls = num_class

        # weight matrix init
        self.weight_matrix = torch.nn.Parameter(torch.empty(self.num_cls, self.emb_size))
        nn.init.xavier_normal_(self.weight_matrix)

        self.soft_plus = nn.Softplus()

    def forward(self, embedding, label):

        sim = nn.functional.linear(nn.functional.normalize(embedding ,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight_matrix,p=2, dim=1, eps=1e-12))

        index_p = F.one_hot(label, num_classes = self.num_cls).type(dtype = torch.bool)
        index_n = torch.logical_not(index_p)

        # one_hot = torch.zeros(sim.size())
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # one_hot = one_hot.type(dtype=torch.bool)
        # #sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sim_p = sim[index_p]
        # mask = one_hot.logical_not()
        sim_n = sim[index_n]

        # sp = index_p * sim
        # sn = index_n * sim

        sim_p = sim_p.view(embedding.size()[0], -1)
        sim_n = sim_n.view(embedding.size()[0], -1)

        o_p = 1 + self.m
        o_n = -self.m

        delta_p = 1 - self.m
        delta_n = self.m

        alpha_p = o_p - sim_p
        alpha_n = sim_n - o_n

        # ap = o_p - sp
        # an = o_n - sn

        # ap = (ap > 0) * ap
        # an = (an > 0) * an

        alpha_p = (alpha_p > 0) * alpha_p
        alpha_n = (alpha_n > 0) * alpha_n

        # fake_p = - ap * (sp - delta_p) * self.gamma 
        # fake_n = an * (sn - delta_n) * self.gamma 

        # fake_loss = self.soft_plus(torch.logsumexp(fake_n, 1) + torch.logsumexp(fake_p, 1))

        p = - alpha_p * (sim_p - delta_p) * self.gamma
        n = alpha_n * (sim_n - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(n, 1) + torch.logsumexp(p, 1))

        # 밑에 있는 식은 제대로 동작을 안하는데 이유가 뭔지 확인 바람.
        # 원인은 애초에 p와 n에 대한 index가 달라서 아래와 같이 합쳐서 계산해줄 수가 없음.
        # loss = np.log((1+np.sum(np.exp(self.gamma*(n - p)), axis=1)))

        '''
        numpy 버전 구현
        '''

        # sim = torch.mm(embedding, self.weight_matrix)

        # sim_p = np.eye(self.num_cls)[label] * sim
        # sim_n = np.logical_not(sim_p) * sim

        # o_p = 1 + self.m
        # o_n = -self.m

        # delta_p = 1 - self.m
        # delta_n = self.m

        # alpha_p = o_p - sim_p
        # alpha_n = sim_n - o_n

        # alpha_p = (alpha_p > 0) * alpha_p
        # alpha_n = (alpha_n > 0) * alpha_n

        # p = np.sum(np.exp(p), axis=1)
        # n = np.sum(np.exp(n), axis=1)

        # p = alpha_p * (sim_p - delta_p) * self.gamma
        # n = alpha_n * (sim_n - delta_n) * -self.gamma

        # loss = np.log(p + n + 1)

        return loss.mean()

if __name__ == "__main__":

    # 전체적으로 torch로 다시 짜는 게 좋을 듯.
    # 그리고 나서 비교하기. (https://github.com/xialuxi/CircleLoss_Face/blob/master/CircleLoss.py)

    emb = torch.rand(64, 64)
    label = torch.randint(high=9, size=(64,))

    loss = circle_loss()

    print(loss(emb, label))
    


#from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py 

from typing import Tuple
import torch.nn.functional as F

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
    
class CircleLossTriplet(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='dot'):
        super(CircleLossTriplet, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity
        
    def forward(self, p, n, q):
        if self.similarity == 'dot':
            sim_p = torch.matmul(q, p.t())
            sim_n = torch.matmul(q, n.t())
        elif self.similarity == 'cos':
            sim_p = F.cosine_similarity(q, p)
            sim_n = F.cosine_similarity(q, n)
        else:
            raise ValueError('This similarity is not implemented.')
            
        alpha_p = F.relu(-sim_p + 1 + self.margin)
        alpha_n = F.relu(sim_n + self.margin)
        margin_p = 1 - self.margin
        margin_n = -self.margin
        
        logit_p = self.scale * alpha_p * (sim_p - margin_p)
        logit_n = self.scale * alpha_n * (sim_n - margin_n)
        
        label_p = torch.ones_like(logit_p)
        label_n = torch.zeros_like(logit_n)
        
        loss = F.binary_cross_entropy_with_logits(torch.cat([logit_p, logit_n]), 
                                                  torch.cat([label_p, label_n]))
        return loss.mean()



if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)

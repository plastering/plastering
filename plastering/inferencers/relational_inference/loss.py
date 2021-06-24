import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class tripletLoss(nn.Module):
    def __init__(self, margin):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        loss = F.relu(distance_pos - distance_neg + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


class angularLoss(nn.Module):
    def __init__(self, margin, l=1):
        super(angularLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1).pow(1 / 2)
        distance_cen = (neg - anchor * 0.5 - pos * 0.5).pow(2).sum(1)
        loss = F.relu(distance_pos - self.l * distance_cen + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


class softmaxtripletLoss(nn.Module):
    def __init__(self):
        super(softmaxtripletLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, anchor, pos, neg):
        n = anchor.size(0)
        d2pos = self.dist(anchor, pos)
        d2neg = self.dist(anchor, neg)
        # print("pos neg", d2pos,"\n", d2neg)
        e_pos = torch.exp(d2pos)
        e_neg = torch.exp(d2neg)
        # print("e_pos", "e_neg:", e_pos, e_neg)
        d_pos = e_pos / (e_pos + e_neg)
        d_neg = e_neg / (e_pos + e_neg)
        # print("d_pos, d_neg", d_pos, d_neg)
        loss = torch.sum(d_pos ** 2)
        return loss, (d2pos < d2neg).sum()

    def dist(self, a, b):  # dim = -1
        d = a - b
        # print(d)
        d = d ** 2
        # print(d)
        d = self.relu(d)
        return torch.sqrt(torch.sum(d, dim=-1))


class combLoss(nn.Module):
    def __init__(self, margin, l = 1):
        super(combLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        distance_cen = (neg - anchor * 0.5 - pos * 0.5).pow(2).sum(1)
        loss = F.relu(distance_pos - self.l * distance_cen + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()

def main():
    loss = angularLoss(margin=1)

    anchor = torch.Tensor([[1, 2], [1, 2]])
    pos = torch.Tensor([[1, 3], [2, 2]])
    neg = torch.Tensor([[0, 1], [3, 4]])
    out_ = loss(anchor, pos, neg)
    print(out_)


if __name__ == '__main__':
    main()

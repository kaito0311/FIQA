import torch
from torch import nn

import math
import numpy as np


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CR_FIQA_LOSS(nn.Module):
    r"""Implement of ArcFace:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(CR_FIQA_LOSS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[
                            1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        with torch.no_grad():
            distmat = cos_theta[index, label.view(-1)].detach().clone()
            max_negative_cloned = cos_theta.detach().clone()
            max_negative_cloned[index, label.view(-1)] = -1e-12
            max_negative, _ = max_negative_cloned.max(dim=1)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, 0, distmat[index, None], max_negative[index, None]


class CR_FIQA_LOSS_ONTOP():
    r"""Implement of ArcFace:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, path_mean, path_std, device="cuda", s=64.0, m=0.50):
        self.device = device
        self.s = s
        self.m = m
        self.mean = np.load(path_mean).astype(np.float32)
        self.kernel = torch.from_numpy(self.mean.T).to(self.device)
        self.std = np.load(path_std).astype(np.float32)
        self.std = torch.from_numpy(self.std.T).to(self.device)

        print("[INFO] size kernel: ", self.kernel.size())
        # nn.init.normal_(self.kernel, std=0.01)

    def __call__(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        embbedings = (embbedings - self.kernel.T[label]) / self.std.T[label]
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        # m_hot = torch.zeros(index.size()[0], cos_theta.size()[
        #                     1], device=cos_theta.device)
        # m_hot.scatter_(1, label[index, None], self.m)
        distmat = cos_theta[index, label.view(-1)].detach().clone()
        max_negative_cloned = cos_theta.detach().clone()
        max_negative_cloned[index, label.view(-1)] = -1e-12
        max_negative, _ = max_negative_cloned.max(dim=1)
        max_negative = max_negative.clamp(-1, 1)
        # print(distmat[:10])
        # print(max_negative[:10])
        divi = (distmat/(max_negative+1+1e-9))
        # print(divi[:10])
        sub = distmat - max_negative
        index_neq = torch.where(sub < 0)[0]
        # cos_theta.acos_()
        # cos_theta[index] += m_hot
        # cos_theta.cos_().mul_(self.s)
        return cos_theta, index_neq, distmat[index, None], max_negative[index, None]


class CR_FIQA_LOSS_COSINE:
    def __init__(self, path_mean_feature, path_list_mean_cosine, path_list_std_cosine, device="cuda", s=64.0, m=0.50):
        self.device = device
        self.s = s
        self.m = m

        self.mean = np.load(path_mean_feature).astype(np.float32)
        self.kernel = torch.from_numpy(self.mean.T).to(self.device)

        self.mean_cosine_dis = np.load(
            path_list_mean_cosine).astype(np.float32)
        self.mean_cosine_dis = torch.from_numpy(
            self.mean_cosine_dis).to(self.device)

        self.std_cosine_dis = np.load(path_list_std_cosine).astype(np.float32)
        self.std_cosine_dis = torch.from_numpy(
            self.std_cosine_dis.T).to(self.device)

        print("[INFO] size kernel: ", self.kernel.size())

    def __call__(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        
        cos_theta_norm = (
            cos_theta - self.mean_cosine_dis.reshape(1, -1)) / self.std_cosine_dis.reshape(1, -1)
        cos_theta = cos_theta_norm

        index = torch.where(label != -1)[0]
        distmat = cos_theta[index, label.view(-1)].detach().clone()

        max_negative_cloned = cos_theta.detach().clone()
        max_negative_cloned[index, label.view(-1)] = -1000
        max_negative, _ = max_negative_cloned.max(dim=1)

        return cos_theta, None, distmat[index, None], max_negative[index, None]


def smooth_l1_loss(x, y, scale_loss=None, beta=0.5):
    sub = torch.abs(x - y)

    score = torch.where(sub < beta, 0.5 * sub **
                        2 / beta, sub - 0.5 * beta)

    if scale_loss is not None:
        score = score * scale_loss

    return torch.mean(score)
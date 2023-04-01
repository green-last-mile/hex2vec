import torch
from torch import nn
import torch.nn.functional as F
from torch import sigmoid
import pytorch_lightning as pl
from typing import List, Tuple
from torchmetrics.functional import f1_score as f1


# TODO! This is incomplete


class GeoVeXZIP(nn.Module):
    def __init__(self):
        super(GeoVeXZIP, self).__init__()

    def forward(self, g_pi, g_lambda):
        pi = torch.sigmoid(g_pi)
        lambda_ = torch.exp(g_lambda)

        return pi, lambda_


class GeoVeXDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GeoVeXDecoder, self).__init__()

        self.zip_layer = GeoVeXZIP()
        self.fc_pi = nn.Linear(input_size, output_size)
        self.fc_lambda = nn.Linear(input_size, output_size)

    def forward(self, h):
        g_pi = self.fc_pi(h)
        g_lambda = self.fc_lambda(h)

        pi, lambda_ = self.zip_layer(g_pi, g_lambda)

        return pi, lambda_
    

class GeoVeXLoss(nn.Module):
    def __init__(self, R):
        super(GeoVeXLoss, self).__init__()
        self.R = R

    def forward(self, pi, lambda_, y):
        M = 2 * self.R + 1
        K = pi.size(-1)

        def w_dist(i, j):
            r = max(abs(i - self.R), abs(j - self.R), abs(i - j))
            return 1 / (1 + r) if r <= self.R else 0

        def w_num(i, j):
            r = max(abs(i - self.R), abs(j - self.R), abs(i - j))
            return 1 / (6 * r) if r <= self.R and r > 0 else 1 if r == 0 else 0

        w_dist_matrix = torch.tensor([[w_dist(i, j) for j in range(M)] for i in range(M)], dtype=torch.float32)
        w_num_matrix = torch.tensor([[w_num(i, j) for j in range(M)] for i in range(M)], dtype=torch.float32)

        I0 = (y == 0).float()
        I_greater_0 = (y > 0).float()

        log_likelihood_0 = I0 * torch.log(pi + (1 - pi) * torch.exp(-lambda_))
        log_likelihood_greater_0 = I_greater_0 * (torch.log(1 - pi) - lambda_ + y * torch.log(lambda_) - torch.lgamma(y + 1))

        log_likelihood = log_likelihood_0 + log_likelihood_greater_0
        loss = -torch.sum(log_likelihood * w_dist_matrix * w_num_matrix) / (torch.sum(w_dist_matrix) * torch.sum(w_num_matrix))

        return loss
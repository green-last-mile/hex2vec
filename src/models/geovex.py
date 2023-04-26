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

    def forward(self, x):
        pi = torch.sigmoid(x)
        lambda_ = torch.exp(x)
        return pi, lambda_


# class GeoVeXDecoder(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(GeoVeXDecoder, self).__init__()

#         self.zip_layer = GeoVeXZIP()
#         self.fc_pi = nn.Linear(input_size, output_size)
#         self.fc_lambda = nn.Linear(input_size, output_size)

#     def forward(self, h):
#         g_pi = self.fc_pi(h)
#         g_lambda = self.fc_lambda(h)

#         pi, lambda_ = self.zip_layer(g_pi, g_lambda)

#         return pi, lambda_

def build_mask_funcs(R):
    def w_dist(i, j):
        r = max(abs(i - R), abs(j - R), abs(i - j))
        return 1 / (1 + r) if r <= R else 0
    
    def w_num(i, j):
        r = max(abs(i - R), abs(j - R), abs(i - j))
        return 1 / (R * r) if r <= R and r > 0 else 1 if r == 0 else 0

    return w_dist, w_num

class GeoVeXLoss(nn.Module):
    def __init__(self, R):
        super(GeoVeXLoss, self).__init__()
        self.R = R
        self._w_dist, self._w_num = build_mask_funcs(self.R)

        M = 2 * self.R + 1
        self._w_dist_matrix = torch.tensor(
            [[self._w_dist(i, j) for j in range(M)] for i in range(M)], dtype=torch.float32
        )
        self._w_num_matrix = torch.tensor(
            [[self._w_num(i, j) for j in range(M)] for i in range(M)], dtype=torch.float32
        )

    def forward(self, pi, lambda_, y):
        M = 2 * self.R + 1
        K = pi.size(1)

        I0 = (y == 0).float()
        I_greater_0 = (y > 0).float()

        log_likelihood_0 = I0 * torch.log(pi + (1 - pi) * torch.exp(-lambda_))
        log_likelihood_greater_0 = I_greater_0 * (
            torch.log(1 - pi) - lambda_ + y * torch.log(lambda_) - torch.lgamma(y + 1)
        )

        log_likelihood = log_likelihood_0 + log_likelihood_greater_0
        loss = -torch.sum(log_likelihood * self._w_dist_matrix * self._w_num_matrix) / (
            torch.sum(self._w_dist_matrix) * torch.sum(self._w_num_matrix)
        )

        return loss


class HexagonalConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
    ):
        super(HexagonalConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.register_buffer("hexagonal_mask", self.create_hexagonal_mask())

    def create_hexagonal_mask(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return mask

    def forward(self, x):
        self.conv.weight = nn.Parameter(self.conv.weight * self.hexagonal_mask)
        out = self.conv(x)
        return out


class HexagonalConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        super(HexagonalConvTranspose2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=bias,
        )
        self.register_buffer("hexagonal_mask", self.create_hexagonal_mask())

    def create_hexagonal_mask(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return mask

    def forward(self, x):
        self.conv_transpose.weight.data *= self.hexagonal_mask
        out = self.conv_transpose(x)
        return out


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class GeoVexModel(pl.LightningModule):
    def __init__(self, k_dim, R, lr=1e-3, weight_decay=1e-5):
        super().__init__()

        self.k_dim = k_dim
        self.R = R
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(self.k_dim),
            nn.ReLU(),
            HexagonalConv2d(self.k_dim, 256, kernel_size=3),
            # HexagonalConv2d(512, 256, kernel_size=3),
            HexagonalConv2d(256, 128, kernel_size=3),
            nn.Flatten(),
            # TODO: make this a function of R
            nn.Linear(5 * 5 * 128, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 5 * 5 * 128),
            # maintain the batch size, but reshape the rest
            Reshape((-1, 128, 5, 5)),
            HexagonalConvTranspose2d(128, 256, kernel_size=3),
            HexagonalConvTranspose2d(256, self.k_dim, kernel_size=3),
            GeoVeXZIP(),
        )

        self._loss = GeoVeXLoss(self.R)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        pi, lambda_ = self(x)
        loss = self._loss.forward(pi, lambda_, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x  = batch
        pi, lambda_ = self(x)
        loss = self._loss.forward(pi, lambda_, x)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

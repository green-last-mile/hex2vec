import torch
from torch import nn
import torch.nn.functional as F
from torch import sigmoid
import pytorch_lightning as pl
from typing import List, Tuple
from torchmetrics.functional import f1_score as f1


class BinaryNN(pl.LightningModule):
    def __init__(self, encoder_sizes):
        super().__init__()

        def create_layers(sizes: List[Tuple[int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        sizes = list(zip(encoder_sizes[:-1], encoder_sizes[1:]))
        self.encoder = create_layers(sizes)

    def forward(self, Xt: torch.Tensor, Xc: torch.Tensor):
        Xt_em = self.encoder(Xt)
        Xc_em = self.encoder(Xc)
        return torch.mul(Xt_em, Xc_em).sum(dim=1)

    def predict(self, Xt: torch.Tensor, Xc: torch.Tensor):
        return sigmoid(self(Xt, Xc))

    def training_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos = self(Xt, Xc)
        scores_neg = self(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos = self(Xt, Xc)
        scores_neg = self(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(sigmoid(scores), y.int(), task="binary")
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_f1", f_score, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DistanceNN(pl.LightningModule):
    def __init__(self, encoder_sizes, max_distance=10):
        super().__init__()

        self.max_distance = max_distance

        def create_layers(sizes: List[Tuple[int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes) - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        sizes = list(zip(encoder_sizes[:-1], encoder_sizes[1:]))
        self.encoder = create_layers(sizes)

        self.distance = nn.Sequential(
            nn.Linear(encoder_sizes[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, Xt: torch.Tensor, Xc: torch.Tensor):
        emb1 = self.encoder(Xt)
        emb2 = self.encoder(Xc)
        distance_vector = torch.abs(emb1 - emb2)
        distance = self.distance(distance_vector)
        return distance
        
    def predict(self, Xt: torch.Tensor, Xc: torch.Tensor):
        return sigmoid(self(Xt, Xc))

    def custom_weighted_mse_loss(self, y_true, y_pred, weight_factor=1.0):
        # Apply a weight to the error proportional to the distance
        y_pred_clamped = torch.clamp(y_pred, 0, self.max_distance)
        # make the prediction an integer

        weights = torch.pow(y_true, -weight_factor)
        # Calculate the mean squared error
        mse_loss = torch.mean(weights * torch.pow(y_true - y_pred, 2))
        return mse_loss

    def training_step(self, batch, batch_idx):
        Xt, Xc, y_act, *_ = batch
        predicted_distance = self(Xt, Xc)
        loss = self.custom_weighted_mse_loss(y_act, predicted_distance)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Xt, Xc, y_act, *_ = batch
        predicted_distance = self(Xt, Xc)
        loss = self.custom_weighted_mse_loss(y_act, predicted_distance)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

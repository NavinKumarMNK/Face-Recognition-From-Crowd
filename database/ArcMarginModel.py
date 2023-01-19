import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Parameter
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer

class ArcMarginModel(LightningModule):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()
        self.embedding_size = args.embedding_size
        self.num_classes = args.num_classes
        self.margin = args.margin
        self.embedding = nn.Linear(self.embedding_size, self.num_classes).double()

    def forward(self, embeddings):
        # pass the embeddings through the linear layer
        embeddings
        logits = self.embedding(embeddings)
        return logits

    def contrastive_loss(self, logits, labels):
        # calculate the similarity between embeddings
        similarity = torch.matmul(logits, logits.t())
        diagonal = similarity.diag().view(-1, 1)
        cost_s = torch.clamp(self.margin - diagonal + similarity, min=0)
        cost_im = torch.clamp(diagonal + similarity - 2 * self.margin, min=0)
        mask = torch.eye(similarity.size(0), device="cuda") > .5
        I = torch.eye(similarity.size(0), device="cuda", dtype=torch.bool)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        cost_s = cost_s.max()
        cost_im = cost_im.max()
        cost_im = torch.mul(cost_im, labels)
        cost = cost_s + cost_im
        return cost

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def prediction_step(self, batch):
        embeddings = batch
        logits = self.forward(embeddings)
        predictions = logits.argmax(1)
        return {"predictions": predictions}


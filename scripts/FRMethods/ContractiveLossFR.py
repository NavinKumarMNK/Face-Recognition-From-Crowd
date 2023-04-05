'@Author: NavinKumarMNK' 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils import utils
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContractiveLossFR(LightningModule):
    def __init__(self, embedding_size:int, num_classes:int, 
                    margin:float, easy_margin:bool, pretrained:bool):
        super(ContractiveLossFR, self).__init__()
        self.embedding_size = int(embedding_size)
        self.num_classes = num_classes
        self.margin = float(margin)
        self.easy_margin = easy_margin
        self.embedding = nn.Sequential(
                nn.Linear(self.embedding_size, 
                                self.embedding_size // 2),
                nn.BatchNorm1d(self.embedding_size // 2),
                nn.GELU(),
                nn.Linear(self.embedding_size // 2, 
                                    self.embedding_size // 4),
                nn.BatchNorm1d(self.embedding_size // 4),
                nn.GELU(),
                nn.Linear(self.embedding_size // 4, 
                        self.embedding_size // 8),
                nn.BatchNorm1d(self.embedding_size // 8),
                nn.GELU(),
                nn.Linear(self.embedding_size // 8, 
                        self.num_classes)
            )
        try:
            if(pretrained == True):
                path = utils.ROOT_PATH + '/weights/contractive_loss_weights.pt'
                self.embedding = torch.load(path)      
        except Exception as e:
            torch.save(self.embedding, path)
        self.embedding.to(DEVICE)

    def forward(self, embeddings):
       
        dtype = next(self.embedding.parameters()).dtype
        print(dtype)
    
        logits = self.embedding(embeddings)
        return logits

    def contrastive_loss(self, logits, labels):
        # calculate the similarity between embeddings
        similarity = torch.norm(logits, dim=1, p=2)
        diagonal = similarity.diag()
        cost_s = torch.clamp(self.margin - diagonal + similarity, min=0)
        mask = torch.eye(similarity.size(0), device=DEVICE) > .5
        I = torch.eye(similarity.size(0), device=DEVICE, dtype=torch.bool)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost = cost_s.max()
        return cost

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def prediction_step(self, batch):
        embeddings = batch
        logits = self.forward(embeddings)
        predictions = logits.argmax(1)
        return {"predictions": predictions}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def on_train_end(self) -> None:
        torch.save(self.embedding, utils.ROOT_PATH + '/weights/contractive_loss_weights.pt')

class ContractiveLossFREmbeddingsDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self._make_dataset()

    def _make_dataset(self):
        df = pd.read_csv(self.file_path)
        self.labels = df.iloc[:, 0].values
        self.embeddings = df.iloc[:, 1:].values
        # convert to float tensor
        self.labels = self.labels.astype(np.float32)
        self.embeddings = self.embeddings.astype(np.float32)

        # from numpy array
        self.labels = torch.from_numpy(self.labels)
        self.embeddings = torch.from_numpy(self.embeddings)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.embeddings[idx], self.labels[idx]

class ContractiveLossFREmbeddingsDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 1):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = ContractiveLossFREmbeddingsDataset(self.file_path)
        # split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, 
                                [int(len(self.dataset)*0.8), int(len(self.dataset)*0.1), 
                                int(len(self.dataset)) - int(len(self.dataset)*0.8) - int(len(self.dataset)*0.1)])        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=6, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=6, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                           shuffle=False, num_workers=6, drop_last=True)

    
if __name__ == "__main__":
    args = utils.config_parse('CONTRACTIVE_LOSS_FR')
    args['num_classes'] = len(os.listdir(utils.ROOT_PATH + '/database/faces')) 
    model = ContractiveLossFR(**args, pretrained=True)
    data = ContractiveLossFREmbeddingsDataModule(utils.ROOT_PATH + '/database/embeddings.csv', batch_size=1)
    data.prepare_data()
    trainer = pl.Trainer(
    accelerator='gpu', devices=1,
    min_epochs=10,
    max_epochs=25,
    )
    trainer.fit(model, data)
    trainer.save_checkpoint(utils.ROOT_PATH + '/weights/contractive_loss.cpkt')

import pytorch_lightning as pl
import sys
import torch
from ArcMarginModel import ArcMarginModel
from torch.utils.data import DataLoader, random_split
import os
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
# number of folders in folder faces
args = type("args", (object,), {})()
args.num_classes = len(os.listdir("./faces"))
args.margin_m = 0.2
args.margin_s = 64
args.emb_size = 512
args.easy_margin = False

# pl datamodule read data from json

class FaceEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class FaceEmbeddingsDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        self.labels = df.iloc[:, 0].values
        self.embeddings = df.iloc[:, 1:].values
        # convert to numpy array
        self.labels =   torch.Tensor(self.labels).long()
        self.embeddings = torch.tensor(self.embeddings)
        self.train_embeddings, self.val_embeddings, self.train_labels, self.val_labels = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
    
    def train_dataloader(self):
        train_dataset =  FaceEmbeddingsDataset(self.train_embeddings, self.train_labels)
        return DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset =  FaceEmbeddingsDataset(self.val_embeddings, self.val_labels)
        return DataLoader(val_dataset, batch_size=1, num_workers=4)



model = ArcMarginModel(args)
data = FaceEmbeddingsDataModule("./embeddings.csv")
data.prepare_data()
pl.Trainer(
    gpus=1,
    min_epochs=10,
    max_epochs=25,
).fit(model, data)

# save model
torch.save(model.state_dict(), "../weights/arc_margin.pt")
import pytorch_lightning as pl
import sys
import torch
from ArcMarginModel import ArcMarginModel
import os
import pytorch_lightning as pl

# number of folders in folder faces
args = type("args", (object,), {})()
args.num_classes = len(os.listdir("./faces"))
args.margin_m = 0.2
args.margin_s = 64
args.emb_size = 512
args.easy_margin = False

# pl datamodule read data from json

class FacesDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.path = "./embeddings.json"
        self.batch_size = 32
        self.num_workers = 4
        
    def setup(self, stage=None):
        # read json file
        import json
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.data = self.data["embeddings"]
        self.data = [(x["embedding"], x["label"]) for x in self.data]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

model = ArcMarginModel(args)
pl.Trainer(
    gpus=1,
    min_epochs=1,
    max_epochs=10,
).fit(model, FacesDataModule())

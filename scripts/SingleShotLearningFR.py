'@Author: NavinKumarMNK'
import sys 
if '../../' not in sys.path:
    sys.path.append('../../')
    
from utils import utils
import os
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
from scripts.FaceNet import FaceNet
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleShotLearningFR(LightningModule):
    def __init__(self, embedding_size=512, pretrained: bool = True):
        super(SingleShotLearningFR, self).__init__()
        self.facenet = FaceNet(model='resnet101', pretrained=True, im_size=64)
        self.embedding_size = embedding_size
        try:
            if(pretrained == True):
                path = utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt'
                self.load_state_dict(torch.load(path))        
        except Exception as e:
            print("Pretrained weights not found")
                        
    def forward(self, x):
        x = self.facenet(x)
        return x

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + margin)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_end(self) -> None:
        torch.save(self.state_dict(), utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt')

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label_name in os.listdir(self.root_dir):
            label = self.label_map[label_name]
            for image_name in os.listdir(os.path.join(self.root_dir, label_name)):
                image_path = os.path.join(self.root_dir, label_name, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class SSLFacentDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, label_map):
        super(SSLFacentDataModule, self).__init__()
        self.root_dir = root_dir
        self.label_map = label_map

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        transform_train = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = TripletFaceDataset(self.root_dir, self.label_map, transform_train)
        transform_val = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_dataset = TripletFaceDataset(self.root_dir, self.label_map, transform_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    model = SingleShotLearningFR(pretrained=True)
    data = SSLFacentDataModule()
    data.prepare_data()
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, data)
    trainer.save_checkpoint(utils.ROOT_PATH + '/weights/ssl_facenet.ckpt')

    # Yet to complete
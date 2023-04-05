'@Author: NavinKumarMNK'
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from utils import utils
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import json
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
from scripts.FaceNet import FaceNet
from PIL import Image
import pprint as pp
from typing import List
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleShotLearningFR(LightningModule):
    def __init__(self, embedding_size=512, pretrained: bool = True):
        super(SingleShotLearningFR, self).__init__()
        self.facenet = FaceNet(model='resnet101', pretrained=True, im_size=64)
        self.embedding_size = embedding_size
        try:
            if(pretrained == True):
                path = utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt'
                self.facenet = torch.load(path)        
        except Exception as e:
            print("Pretrained weights not found, Saving Trained weights to: ", path)
            torch.save(self.facenet, path)

                        
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
    
    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        #print(anchor.shape, positive.shape, negative.shape)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def on_train_end(self) -> None:
        torch.save(self.facenet, utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt')

class TripletFaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.transform = transform
        self.num_of_permute = 100
        self.image_paths = image_paths
        self.labels = labels
        # calulate number of unique classes
        self.num_classes = len(np.unique(self.labels))
        self.num_image_each_class = {}
        for i in self.labels:
            if i in self.num_image_each_class:
                self.num_image_each_class[i] += 1
            else:
                self.num_image_each_class[i] = 1
        self.train_data_prev: List[List[str]] = []
        self.train_data_after: List[List[str]] = []
        self.process_data()

    def process_data(self):
        """
        @image_paths: list of image paths
        @labels: list of labels
        @output : list of tuples (anchor_image, positive_image, negative_image)
            train_data_prev = [
                    [img1, img2, ...]  # featuring images of class 1
                    [img1, img2, ...]  # featuring images of class 2
            ]
            train_data_prev[0][3] = image
            train_data_after = [
                    [img1, img2, img3]  
                    [img1, img2, img3]  
            ]
        """
        self.labels, self.image_paths = zip(*sorted(zip(self.labels, self.image_paths)))

        temp_label = []
        for label, image in zip(self.labels, self.image_paths):
            if label in temp_label:
                self.train_data_prev[label].append(image)
            else:
                temp_label.append(label)
                self.train_data_prev.append([image])
            
        self.num_image_each_class = dict(sorted(self.num_image_each_class.items(), key=lambda item: item[0]))
        for h in range(self.num_of_permute):
            np.random.seed(int(time.time())) 
            for i in range(self.num_classes):
                for j in range(self.num_image_each_class[i]):
                    a = np.random.randint(0, self.num_image_each_class[i])
                    try:
                        p = (a + np.random.randint(1, self.num_image_each_class[i])) % self.num_image_each_class[i]
                    except Exception as e:
                        print(e)
                        p = a
                    n = 0
                    while n < self.num_classes:
                        if n == i:
                            n+=1
                            continue
                        m = np.random.randint(0, self.num_image_each_class[n])
                        '''print(i, a)
                        print(i, p)
                        print(n, m)
                        print()'''
                        self.train_data_after.append(
                            (self.train_data_prev[i][a], 
                            self.train_data_prev[i][p], 
                            self.train_data_prev[n][m]
                            ))
                        n+=1
        
        del self.train_data_prev
 
    def __len__(self):
        return len(self.train_data_after)

    def __getitem__(self, idx):
        anchor_image = Image.open(self.train_data_after[idx][0])
        if self.transform:
            anchor_image = self.transform(anchor_image)

        positive_image = Image.open(self.train_data_after[idx][1])
        if self.transform:
            positive_image = self.transform(positive_image)


        negative_image = Image.open(self.train_data_after[idx][2])
        if self.transform:
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

class SSLFacentDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, label_map, batch_size=32):
        super(SSLFacentDataModule, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.label_map = label_map
        self.count = 0
        self.transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = []
        self.labels = []
        
    def setup(self, stage=None):
        with open(self.label_map) as f:
            label_map = json.load(f)
        for label_name in os.listdir(self.root_dir):
            label = label_map[label_name]
            label_dir = os.path.join(self.root_dir, label_name)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)
        
        self.triplet_dataset = TripletFaceDataset(self.image_paths, self.labels, self.transform_train)
        
        # split dataset
        train_size = int(0.8 * len(self.triplet_dataset))
        test_size = int(0.1 * len(self.triplet_dataset))
        val_size = len(self.triplet_dataset) - train_size - test_size
        
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            self.triplet_dataset, [train_size, test_size, val_size]
            )
        
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
    model = SingleShotLearningFR(pretrained=True)
    data = SSLFacentDataModule(root_dir=utils.ROOT_PATH+ "/database/faces",
                            label_map=utils.ROOT_PATH+'/database/label_map.json',
                            batch_size=1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
    trainer.fit(model, data)
    trainer.save_checkpoint(utils.ROOT_PATH + '/weights/ssl_facenet.ckpt')

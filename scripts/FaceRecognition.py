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
from FaceNet import FaceNet
from SingleShotLearningFR import SingleShotLearningFR
from ContractiveLossFR import ContractiveLossFR
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceRecognition():
    def __init__(self, recognizer: str = 'ssl'):
        self.recognizer = recognizer
        
        if self.recognizer == 'ssl':
            self.model = SingleShotLearningFR(pretrained=True)
        elif self.recognizer == 'contactive':
            args = utils.config_parse('CONTRACTIVE_LOSS_FR')
            args['num_classes'] = len(os.listdir(utils.ROOT_PATH + '/database/faces')) 
            self.model = ContractiveLossFR(**args, pretrained=True) 
            
        else:
            raise Exception("Invalid recognizer")
        
        self.model = self.model.to(DEVICE)
    
    
        
    def predict(self, image):
        image = Image.open(image)
        image = image.resize((64, 64))
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)
        with torch.no_grad():
            embedding = self.model(image)
        return embedding
    
    def train(self, path):
        pass

if __name__ == '__main__':
    face_recognizer = FaceRecognition(recognizer='contactive')
    #face_recognizer.predict('../../database/faces/Navin/Navin_1.jpg')
    #face_recognizer.train('')
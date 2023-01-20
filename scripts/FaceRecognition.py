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
from scripts.SingleShotLearningFR import SingleShotLearningFR
from scripts.ContractiveLossFR import ContractiveLossFR
from PIL import Image
from scripts.Upsample import Upsample
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceRecognition():
    def __init__(self, recognizer: str = 'ssl'):
        self.recognizer = recognizer
        
        if self.recognizer == 'ssl':
            self.model = SingleShotLearningFR(pretrained=True)
        elif self.recognizer == 'contractive':
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
        #image = Upsample(image)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)
        if(self.recognizer == 'ssl'):
            return self._predict_ssl(image)
        elif(self.recognizer == 'contractive'):
            return self._predict_contractive(image)
        
    def _predict_ssl(self, image):
        return self.model(image)

    def _predict_contractive(self, image):
        with torch.no_grad():
            embedding = self.recognizer(image)
            prediction = self.model(embedding)
            return prediction
    
    def train(self, path):
        pass

if __name__ == '__main__':
    face_recognizer = FaceRecognition(recognizer='contactive')
    #face_recognizer.predict('../../database/faces/Navin/Navin_1.jpg')
    #face_recognizer.train('')
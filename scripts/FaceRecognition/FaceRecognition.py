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
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = type("args", (object,), {})()
    args.num_classes = len(os.listdir("./database/faces"))
    args.margin = 0.2
    args.embedding_size = 512

    class HParams:
        def __init__(self):
            self.pretrained = False
            self.use_se = True

    config = HParams()

    # load state dicts of two models
    model1 = ContractiveLossFR(args)
    state_dict1 = torch.load('./weights/arc_margin.pt', map_location="cuda")
    model1.load_state_dict(state_dict1)

    model2 = resnet101(config)
    state_dict2 = torch.load('./weights/facenet.pt', map_location="cuda")
    model2.load_state_dict(state_dict2)
    model2.fc = nn.Linear(8192, 512)
    model2 = nn.DataParallel(model2)

    # combine the models using nn.Sequential
    combined_model = nn.Sequential(model2, model1)
    combined_model = combined_model.to("cuda")

    # test the combined model
    combined_model.eval()
    inputs = torch.randn(1, 3, 64, 64)
    outputs = combined_model(inputs)
    print(outputs)

    # save the combined model
    print(combined_model)
    torch.save(combined_model.state_dict(), './weights/face_recognize.pt')
    
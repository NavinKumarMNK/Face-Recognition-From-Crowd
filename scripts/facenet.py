import torch
import torch.nn as nn
import sys
sys.path.append("../")
from models.facenet import resnet101
import os
from database.ArcMarginModel import ArcMarginModel

args = type("args", (object,), {})()
args.num_classes = len(os.listdir("../database/faces"))
args.margin_m = 0.2
args.margin_s = 64
args.emb_size = 512
args.easy_margin = False

class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

config = HParams()

# load state dicts of two models
model1 = ArcMarginModel(args)
state_dict1 = torch.load('../weights/arc_margin.pt', map_location="cuda")
model1.load_state_dict(state_dict1)

model2 = resnet101(config)
state_dict2 = torch.load('../weights/facenet.pt', map_location="cuda")
model2.load_state_dict(state_dict2)

# combine the models using nn.Sequential
combined_model = nn.Sequential(model1, model2)

# test the combined model
inputs = torch.randn(1, 3, 64, 64)
outputs = combined_model(inputs)
print(outputs)

# save the combined model
print(combined_model.summary())
torch.save(combined_model.state_dict(), '../weights/face_recognize.pt')

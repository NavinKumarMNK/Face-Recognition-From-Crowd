import torch
import torch.nn as nn
import sys
sys.path.append("../models")
from facenet import resnet101
import os
import uuid
import cv2
import json
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image

transform = transforms.Compose([
    Resize((64, 64)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True
config = HParams()


folders_created = []
while True:
    label = input("Enter Your Label = ")
    if label == "exit":
        break
    #capture the face through cv2 and add create a folder and add the image to the folder
    folder_path = os.path.join("faces", label)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        print("User already exists. Create New User")
        continue
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        #yolov7 to detect face

        a = input("Press Enter to capture q to quit")
        if a == "q":
            break
        
        #save the image to the folder
        image_name = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, frame)

    cap.release()
    cv2.destroyAllWindows()
    folders_created.append(folder_path)

filename = "../weights/facenet.pt"
device = "cuda" if torch.cuda.is_available else "cpu"

model = resnet101(config)
model.load_state_dict(torch.load(filename, map_location=torch.device('cuda')))
model.fc = nn.Linear(8192, 512)
model = nn.DataParallel(model)

for folder in folders_created:
    print("Adding {} to database".format(folder))
    images = os.listdir(folder)
    for image in images:
        image_path = os.path.join(folder, image)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        print(image.shape)
        with torch.no_grad():
            embedding = model(image)
            embedding = embedding.cpu().numpy()
            embedding = embedding.reshape(512)
            embedding = embedding.tolist()
            #add in embedding.json username => embedding
            with open("embeddings.json", "r") as f:
                data = json.load(f)
                data[folder.split("/")[-1]] = embedding
            with open("embeddings.json", "w") as f:
                json.dump(data, f)
    



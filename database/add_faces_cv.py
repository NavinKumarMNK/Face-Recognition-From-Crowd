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
import pandas as pd

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


# add the label name and label id to the label_map.json file
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    label_id = len(label_map)
    for folder in folders_created:
        label = folder.split("/")[-1]
        if label not in label_map:
            label_map[label] = len(label_map)

with open("label_map.json", "w") as f:
    json.dump(label_map, f)


embeddings = []
for folder in folders_created:
    print("Adding {} to database".format(folder))
    images = os.listdir(folder)
    for image in images:
        image_path = os.path.join(folder, image)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        print(image.shape)
        model.eval()
        embedding = model(image)
        embedding = embedding.detach().cpu().numpy()
        embedding = embedding.reshape(512)
        embedding = embedding.tolist()
        embeddings.append([label_id] + embedding)

if os.path.exists("embeddings.csv"):
    df = pd.read_csv("embeddings.csv")
    df = df.concat([df, pd.DataFrame(embeddings, columns=df.columns)])
    df.to_csv("embeddings.csv", index=False)
else:
    columns = ["label"] + ["embedding_"+str(i) for i in range(512)]
    df = pd.DataFrame(embeddings, columns=columns)
    df.to_csv("embeddings.csv", index=False)


    

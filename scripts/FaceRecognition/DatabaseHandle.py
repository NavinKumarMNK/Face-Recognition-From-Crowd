'''
import torch
import torch.nn as nn
import sys
sys.path.append("../models")
from facenet import resnet101
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image
import pandas as pd

class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

config = HParams()

transform = transforms.Compose([
    Resize((64, 64)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

folders_created = []
class FolderHandler(FileSystemEventHandler):
    def __init__(self, str) -> None:
        self.str = str
        self.previous_folders = set(os.listdir(str))

    def on_created(self, event):
        # check if the event is for a directory
        if event.is_directory:
            print("directory created:{}".format(event.src_path))
            current_folders = set(os.listdir(self.str))
            new_folders = current_folders - self.previous_folders
            print(new_folders)
            for folder in new_folders:
                folder_path = os.path.join(self.str, folder)
                folders_created.append(folder_path)
            self.previous_folders = current_folders

observer = Observer()
observer.schedule(FolderHandler('./faces'), path='./faces', recursive=True)
observer.start()


print("Add folders to database/faces .\nEach folder should contain one or more images of a person.\n Name the folder with the person name. \n Press Enter once your are done")
input()

observer.stop()
observer.join()

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

'''
    
    

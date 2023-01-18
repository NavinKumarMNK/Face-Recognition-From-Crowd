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
class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

config = HParams()

folders_created = []
class FolderHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.previous_folders = set()

    def on_created(self, event):
        # check if the event is for a directory
        if event.is_directory:
            print("directory created:{}".format(event.src_path))
            current_folders = set(os.listdir(event.src_path))
            new_folders = current_folders - self.previous_folders
            for folder in new_folders:
                if os.path.isdir(os.path.join(event.src_path, folder)):
                    folder_path = os.path.join(event.src_path, folder)
                    folders_created.append(folder_path)
            self.previous_folders = current_folders

observer = Observer()
observer.schedule(FolderHandler(), path='./faces', recursive=True)
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

for folder in folders_created:
    print("Adding {} to database".format(folder))
    images = os.listdir(folder)
    for image in images:
        image_path = os.path.join(folder, image)
        print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (160, 160))
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, 3, 160, 160)
        image = torch.from_numpy(image).float()
        image = image.to(device)
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
    



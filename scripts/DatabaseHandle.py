'@Author: NavinKumarMNK'
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.FaceNet import FaceNet
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import json
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image
import pandas as pd
import uuid
from utils import utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import subprocess
class DatabaseHandler():
    def __init__(self, database=utils.ROOT_PATH+'/database', weights='/weights/ssl_facenet_weights.pt') -> None:
        self.database = database
        self.weights = weights
        self.model = FaceNet(model='resnet101', pretrained=False, im_size=64)
        self.model = torch.load(utils.ROOT_PATH + self.weights)
        self.model.to(DEVICE)
        self.transform = transforms.Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def add_label_folder(self):
        DatabaseHandler.folders_created = []
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
                        DatabaseHandler.folders_created.append(folder_path)
                    self.previous_folders = current_folders
        
        observer = Observer()
        observer.schedule(FolderHandler(self.database+'/faces'), path=self.database+'/faces', recursive=True)
        observer.start()

        input("Add label-folders to database/faces . Press Enter once your are done")

        observer.stop()
        observer.join()

        self.face2database()

    def update_embeddings(self):
        update_folders = []
        with open(self.database+"/label_map.json", "r") as f:
            label_map = json.load(f)
        for label in label_map:
            label_id = label_map[label]
            update_folders.append(self.database+'/faces/'+label)

        embeddings = []
        label_id = 0
        for folder in update_folders:
            print("Adding {} to database".format(folder))
            images = os.listdir(folder)
            for image in images:
                # change image name to uuid name and save it
                image_path = os.path.join(folder, image)
                image = Image.open(image_path)
                image = self.transform(image).unsqueeze(0)
                print(image.shape)
                image = image.to(DEVICE)
                self.model.eval()
                embedding = self.model(image)
                embedding = embedding.detach().cpu().numpy()
                embedding = embedding.reshape(512)
                embedding = embedding.tolist()
                embeddings.append([label_id] + embedding)
            label_id+=1


        columns = ["label"] + ["embedding_"+str(i) for i in range(512)]
        df = pd.DataFrame(embeddings, columns=columns)
        df.to_csv(self.database+"/embeddings.csv", index=False)    


    def face2database(self):
        with open(self.database+"/label_map.json", "r") as f:
            label_map = json.load(f)
            label_id = len(label_map)
            for folder in self.folders_created:
                label = folder.split("/")[-1]
                if label not in label_map:
                    label_map[label] = len(label_map)

        with open(self.database+"/label_map.json", "w") as f:
            json.dump(label_map, f)

        embeddings = []
        for folder in self.folders_created:
            print("Adding {} to database".format(folder))
            images = os.listdir(folder)
            for image in images:
                # change image name to uuid name and save it
                image_temp = str(uuid.uuid4()) + ".jpg"
                os.rename(os.path.join(folder, image), os.path.join(folder, image_temp))
                image = image_temp
                image_path = os.path.join(folder, image)
                command = "python3 run.py --source " + image_path + " --crop-face"
                process = subprocess.Popen(command, shell=True)
                process.wait()
                image = Image.open(image_path)
                image = self.transform(image).unsqueeze(0)
                print(image.shape)
                image = image.to(DEVICE)
                self.model.eval()
                embedding = self.model(image)
                embedding = embedding.detach().cpu().numpy()
                embedding = embedding.reshape(512)
                embedding = embedding.tolist()
                embeddings.append([label_id] + embedding)
            label_id+=1

        if os.path.exists(self.database+"/embeddings.csv"):
            df = pd.read_csv(self.database+"/embeddings.csv")
            df = pd.concat([df, pd.DataFrame(embeddings, columns=df.columns)])
            df.to_csv(self.database+"/embeddings.csv", index=False)
        else :
            columns = ["label"] + ["embedding_"+str(i) for i in range(512)]
            df = pd.DataFrame(embeddings, columns=columns)
            df.to_csv(self.database+"/embeddings.csv", index=False)    

if __name__ == "__main__":
    db = DatabaseHandler()
    db.add_label_folder()
    #db.update_embeddings()

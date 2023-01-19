import sys, os
sys.path.append(os.path.abspath('../'))
from utils.torch_utils import TracedModel, load_classifier, select_device, time_synchronized
from utils.plots import plot_one_box
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from PIL import Image
from utils.datasets import LoadImages, LoadStreams
from models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import argparse
import asyncio
import numpy as np
from scripts.tracker import *
from utils import utils


class Process():
    def __init__(self, temp_dir, weigths, source, device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.temp_dir = temp_dir
        self.capture = None
        self.weights = weigths
        self.path = source
        self.source = utils.path2src(source)
        if(self.source == 'image'):
            self.weights = self.weights.replace('tinyface.pt', 'face.pt')
        self.device = device
        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    
    def process_frame(self):
        set_logging()
        if(self.device == "cuda"):
            self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, map_location=
                                self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)

        self.model = TracedModel(self.model, self.device, 640)
        if self.half:
            self.model.half()
        
        if (self.source == 'live'):
            for frames in self._process_live(): 
                yield frames    

        elif (self.source == 'image'):
            yield self._process_image(self.path)
        else:
            yield self._process_video()
        
    def _process_live(self):
        self.source = '0'
        
        cudnn.benchmark = True
        self.dataset = LoadStreams(self.source, img_size=640, stride=self.stride)
                                
        
        if self.device != "cpu":
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
        
            if self.device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
                ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=1)[0]
            
            t1 = time_synchronized()
            pred = self.model(img, augment=1)[0]
            print(pred)
            pred = non_max_suppression(pred, 0.25, 0.45, 
                        classes=None, agnostic=False)
            
            t2 = time_synchronized()
            print(pred)

            for i, det in enumerate(pred):
                (p, s, im0, frame) = (path[i], "%g: " % i, 
                                    im0s[i].copy(), self.dataset.count)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                          
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    
                    for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # add to string
                            s += f"{n} {'face'}{'s' * (n > 1)}, "
                
                    dets_to_sort = np.empty((0, 6))
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack(
                            (dets_to_sort, np.array(
                                [x1, y1, x2, y2, conf, detclass]))
                        )

                    tracked_dets = self.sort_tracker.update(
                            dets_to_sort, 1
                        )
                    temp_identities = []
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        if (temp_identities != identities):
                            #save image of face
                            for i, box in enumerate(bbox_xyxy):
                                x1, y1, x2, y2 = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                                face = im0[y1:y2, x1:x2]
                                cv2.imwrite(os.path.join(self.temp_dir, f"{identities[i]}.jpg"), face)
                            temp_identities = identities
                        else:
                            pass
                        confidences = dets_to_sort[:, 4]

                    im0 = self.draw_boxes(
                        im0, bbox_xyxy, identities=identities, categories=categories, 
                            confidences=confidences, names="face", 
                            color=1
                    )

            '''print(
            f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference"
            )'''

            im0 = im0.tobytes()
            yield im0


    def _process_image(self):
        yield frame
    
    def draw_boxes(self, img, bbox, identities=None, categories=0, confidences=None, names=None, color=None):
        faces = []
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            face = img[y1:y2, x1:x2]
            faces.append(face)
            # line thickness
            tl = (round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)

            id = int(identities[i]) if identities is not None else 0
            conf = confidences

            color = 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

            label = (
                str(id) + ":" + "Face" + " - " + str(conf)
                if identities is not None
                else f"{categories} {confidences}"
            )
            # font thickness
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -
                          1, cv2.LINE_AA)
            cv2.putText(
                img,
                label,
                (x1, y1 - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        ret, buffer = cv2.imencode('.jpg', img)
        return buffer


    def _process_video(self):
        self.source = self.path
        view_img = check_imshow()
        self.dataset = LoadImages(self.source, img_size=640, stride=self.stride)

        if self.device != "cpu":
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
    
    def start_capture(self):
        self.count=0
        # capture faces
        if(self.source == 'live'):
            for frame in self.process_frame():
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                #yield frame
        elif(self.source == 'image'):
            print('live3')
            self.process_frame()
        else:
            print('live2')
            self.process_frame()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,)
    args = parser.parse_args()
    for frame in Process('./temp', './weights/yolov7-tinyface.pt', args.source).start_capture():
        print(frame)
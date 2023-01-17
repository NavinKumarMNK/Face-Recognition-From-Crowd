from utils.torch_utils import TracedModel, load_classifier, select_device, time_synchronized
from utils.plots import plot_one_box
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    increment_path,
    non_max_suppression,
    non_max_suppression_lmks,
    scale_coords,
    scale_coords_lmks,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from PIL import Image
from utils.datasets import LoadImages, LoadStreams
from models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
import sys
from pathlib import Path
import time
import argparse
import asyncio
import numpy as np
from scripts.tracker import *
from scripts.facenet import *
from utils import utils

class Process():
    def __init__(self, temp_dir, weigths, device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.temp_dir = temp_dir
        self.capture = None
        self.weights = weigths
        self.device = device
        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    def set_capture(self, source):
        self.path = source
        self.source = utils.path2src(source)
        if (self.source == "live"):
            # set capture to web stream
            self.capture = cv2.VideoCapture(0)
            # set frame rate, height, width
            self.capture.set(cv2.CAP_PROP_FPS, 6)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        elif (self.source == 'image'):
            # no capture just inference on image
            self.capture = None
            pass
        else :
            # set capture on video files
            self.capture = cv2.VideoCapture(source)
        return self

    async def process_frames(self, capture):
        # Initialize
        set_logging()
        print(self.device)
        if (self.device == "cuda"):
            self.device = select_device('0')
        half = self.device.type != "cpu"

        # load model
        model = attempt_load(
            self.weights, map_location=torch.device(self.device))
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)

        model = TracedModel(model, self.device, 640)
        if half:
            model.half()

        # Get names and colors
        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in names]

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.source == "true":
            view_img = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        if (self.source == "live"):
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                else:
                    # write on the frame and send it to the client
                    cv2.putText(
                        frame,
                        "Press 'q' to quit",
                        (20, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        2,
                    )
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')


        # Run inference
        if self.device != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression_lmks(
                pred, 0.25, 0.45, classes=None, agnostic=False)
            t2 = time_synchronized()

            print(len(pred))

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.source == "live":  # batch_size >= 1
                    p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(
                    ), dataset.count
                elif self.source != "image":
                    p, s, im0, frame = path, "", im0s, getattr(
                        dataset, "frame", 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                img_arr = np.array(im0)
                image = Image.fromarray(img_arr)
                image.show()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    det[:, 6:16] = scale_coords_lmks(
                        img.shape[2:], det[:, 6:16], im0.shape
                    ).round()

                    # Print results
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
                        dets_to_sort, 'pink'
                    )
                    tracks = self.sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) == 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        im0, faces = self.draw_boxes(
                            im0, bbox_xyxy, identities, categories, confidences, names, colors
                        )

                # Print time (inference + NMS)
                print(
                    f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
                )

                # Stream results
                if dataset.mode != "image":
                    currentTime = time.time()

                    fps = 1 / (currentTime - startTime)
                    startTime = currentTime
                    cv2.putText(
                        im0,
                        "FPS: " + str(int(fps)),
                        (20, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        2,
                    )

                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

            # write faces
            self.tasks = 0
            for i, face in enumerate(faces):
                cv2.imwrite(f"{self.temp_dir}/{i}.jpg", face)
                task = asyncio.create_task(self.recognize_face(face, i))
                self.tasks.append(task)
            await asyncio.gather(*self.tasks)

        print(f"Done. ({time.time() - t0:.3f}s)")

    async def recognize_face(self, face, i):
        # Running Recognition model
        face_net = FaceNet()
        face_net.set_input(i, face)
        name = face_net.run()
        return name

    def start_capture(self):
        # start async capturing faces
        asyncio.run(self.process_frames(self.capture))

    def draw_boxes(self, img, bbox, identities=None, cat="Face Obtained", confidence=None):
        faces = []
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            face = img[y1:y2, x1:x2]
            faces.append(face)
            # line thickness
            tl = (round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)

            id = int(identities[i]) if identities is not None else 0
            conf = confidence

            color = "pink"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

            label = (
                str(id) + ":" + cat
                if identities is not None
                else f"{cat} {confidence}"
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

        return img, faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,)
    args = parser.parse_args()
    Process('./temp', './weights/yolov7-tinyface.pt',).set_capture(args.source).start_capture()
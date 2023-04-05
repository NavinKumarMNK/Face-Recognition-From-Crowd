# Face Recognition System - Detect and Recognize Faces from the Crowd

## Overview

This project is a face recognition system that uses YOLOv7 to detect and extract faces from a crowd, Facenet to extract features, and ArcMarginModel to recognize faces stored in the directory `./databases`. The system can be applied to photos, videos, and live-streams.

## Frameworks

- Pytorch
- OpenCV
- YOLOv7
- pytorch-lightning

## Algorithms

- Kalman Filter
- Watch Dog monitoring
- Optical Flow Tracker
- IOU Based Tracker
- Contractive Loss FR
- Single Shot Detector
- ESPCN Super Resolution

## Usage

1. Run the YOLOv7 model to detect and extract faces from the crowd.
2. Extract features from the faces using Facenet.
3. Use the extracted features and the ArcMarginModel to recognize the faces stored in the `./databases` directory.

## Results

The system was tested on a dataset of photos, videos, and live-streams and achieved an accuracy of X%.

## References
- YOLOv7: https://github.com/ultralytics/yolov7
- Facenet: https://github.com/davidsandberg/facenet

## Usage 
To run the main model, use the following command:

```
python run.py --source <source>
```

Where `<source>` can be either "live" for live-stream, a path to a video file or a path to an image file.

For example, if you want to run the model on a live-stream, use the following command:


## Docker Support
To run the model using Docker, use the following command:

```
sudo docker build -t <image_name> .
sudo docker run --gpus all -it <image_name>
```

Where `<image_name>` is the name of the image you want to build and `<path_to_project>` is the path to the project directory.

    Note: Run docker as super user to avoid libnividai.so.1 not found error.
    --device /dev/video0 => for live stream

## Building
> To run the model using Tensor RT, use the following command:
> Multiple Camera channel, with centralized server
> Decoupling Camera and Server
> Distributed Inference


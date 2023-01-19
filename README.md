# Face Recognition System - Detect and Recognize Faces from the Crowd
## Overview

This project is a face recognition system that uses YOLOv7 to detect and extract faces from a crowd, Facenet to extract features, and ArcMarginModel to recognize faces stored in the directory `./databases`. The system can be applied to photos, videos, and live-streams.

## Frameworks

- Pytorch
- OpenCV
- YOLOv7
- Facenet
- ArcMarginModel
- pytorch-lightning

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

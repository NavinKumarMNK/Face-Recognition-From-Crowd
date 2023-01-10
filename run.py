import torch
import pytorch_lightning as pt
from pytorch_lightning import Trainer
import argparse
import asyncio
from scripts.process import Process


class Main():
    def __init__(self, args) -> None:
        self.args = args

    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Process(
            './temp', './weights/yolov7-tinyface.pt', device).set_capture(self.args.source).start_capture()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument('--source', type=str,
                        default="video", help="stream/video/image")
    args = parser.parse_args()
    Main(args).run()

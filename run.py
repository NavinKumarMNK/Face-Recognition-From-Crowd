import argparse
import subprocess
import sys
from utils import utils
import os

class Main():
    def __init__(self, args) -> None:
        self.args = args
        self.args.path = args.source
        self.args.source = utils.path2src(args.source)

    def run(self):
        self.initialize()
        if self.args.source == "live": 
            subprocess.run(["python3", "app.py" ])
        else:
            from process import Process
            Process(os.path.abspath('./temp'), os.path.abspath('./weights/yolo-tinyface.pt')).set_capture(self.args.path).start_capture()

    def initialize(self):
        sys.stdout.write("Initializing...\n")
        import __init__
        __init__.init()
        sys.stdout.write("\033[F") 
        sys.stdout.write("Initialized    \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument('--source', type=str,
                        default="video", help="live || video/image path")
    args = parser.parse_args()
    Main(args).run()

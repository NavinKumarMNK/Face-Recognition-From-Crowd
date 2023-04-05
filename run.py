'Author: NavinKumarMNK'
import argparse
import subprocess
import sys
from utils import utils
import os
from process import Process
import cv2
class Main():
    def __init__(self, args) -> None:
        self.args = args
        self.args.path = args.source
        self.args.source = utils.path2src(args.source)
        self.args.temp_dir = "./temp"

    def run(self):
        self.initialize()
        if self.args.source == "live": 
            subprocess.run(["python3", "app.py" ])
        elif (utils.path2src(self.args.path) == "video") :
            #create a folder with name of video and save inside it
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.args.temp_dir+'output.avi', fourcc, 20.0, (640, 480))
            for frame in Process(os.path.abspath('./temp'), 
                    './weights/yolov7-tinyface.pt', self.args.path).start_capture():
                self.out.write(frame)
            self.out.release()
            cv2.destroyAllWindows()

        elif (utils.path2src(self.args.path) == "image") :
            for frame in Process(os.path.abspath(self.args.path), 
                './weights/yolov7-tinyface.pt', self.args.path, crop_faces=True if self.args.crop_face else False).start_capture():
                cv2.imwrite(self.args.path, frame)
        print("Finished !!")

    def initialize(self):
        sys.stdout.write("Initializing...\n")
        import __init__
        __init__.init()
        sys.stdout.write("\033[F") 
        sys.stdout.write("Initialized    \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument('--source', type=str,
                        default="live", help="live || video/image path")
    # crop faces flag with no argument
    parser.add_argument('--crop-face', action='store_true', help="crop faces")
    args = parser.parse_args()
    try:
        Main(args).run()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
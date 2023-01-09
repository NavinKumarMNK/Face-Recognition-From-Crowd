import cv2
import numpy as np


class SourceCapture():
    def __init__(self, source="live", width=640, height=480, fps=30):
        self.onlive = self.onphoto = self.onvideo = False
        if source == "live":
            self.onlive = True
            self.cap = cv2.VideoCapture('0')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        elif source == 'photo':
            self.onphoto = True
        else:
            self.onvideo = True
            self.cap = cv2.VideoCapture(source)

    def _main_live(self, params):
        while True:
            ret, frame = self.cap.read()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if params == 'web_stream':
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                yield frame

    def _main_photo(self, params):
        return cv2.imread(params)

    def _main_video(self, params):
        while True:
            ret, frame = self.cap.read()
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg

    def main(self, params):
        if (self.onlive == True):
            self._main_live(params)
        elif (self.onphoto == True):
            self._main_photo(params)
        else:
            self._main_video(params)


if __name__ == '__main__':
    capture = SourceCapture('photo')
    frame = capture.main("../test/1_1.jpg")
    cv2.imwrite("hello.jpeg", frame)

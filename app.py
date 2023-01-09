import cv2
from flask import Flask, render_template, Response
from scripts.source_capture import SourceCapture

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def main():
    capture = SourceCapture('live')
    capture.main("web_stream")


@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)
